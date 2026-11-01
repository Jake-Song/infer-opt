"""Selected-expert W8A16 kernels for small decode/verification batches."""
import torch
import triton
import triton.language as tl
from torch.library import custom_op, wrap_triton

# Explicit tuning overrides keyed by activation rows. Defaults retain validated arithmetic.
LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}


@triton.jit
def _project(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr, TOP: tl.constexpr,
             GROUP: tl.constexpr, GATE: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(0)
    row = tl.program_id(1)
    expert = tl.load(IDS + route).to(tl.int64)
    token = route // TOP
    col = tl.arange(0, BK)
    acc = tl.full((BK,), 0, tl.float32)
    other = tl.full((BK,), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + col
        x = tl.load(X + (token if GATE else route) * K + k, k < K, 0).to(tl.float32)
        q = tl.load(Q + expert * N * K + row * K + k, k < K, 0).to(tl.float32)
        s = tl.load(S + (expert * N + row) * (K // GROUP) + k // GROUP, k < K, 0).to(tl.float32)
        acc += x * q * s
        if GATE:
            q2 = tl.load(Q + expert * N * K + (row + N // 2) * K + k, k < K, 0).to(tl.float32)
            s2 = tl.load(S + (expert * N + row + N // 2) * (K // GROUP) + k // GROUP, k < K, 0).to(tl.float32)
            other += x * q2 * s2
    value = tl.sum(acc, 0)
    if GATE:
        # Match the projection's BF16 rounding before SwiGLU.
        value = value.to(X.dtype.element_ty).to(tl.float32)
        up = tl.sum(other, 0).to(X.dtype.element_ty).to(tl.float32)
        value = (value * tl.sigmoid(value)).to(X.dtype.element_ty).to(tl.float32) * up
    tl.store(Y + route * (N // 2 if GATE else N) + row, value)


@triton.jit
def _combine(Y, W, OUT, H: tl.constexpr, TOP: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(TOP):
        y = tl.load(Y + (token * TOP + slot) * H + col, col < H, 0).to(tl.float32)
        w = tl.load(W + token * TOP + slot).to(tl.float32)
        weighted = (y * w).to(Y.dtype.element_ty).to(tl.float32)
        acc = (acc + weighted).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(OUT + token * H + col, acc, col < H)


@custom_op('infer_opt::selected_int8_experts', mutates_args=())
def selected_int8_experts(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                          gu: torch.Tensor, gs: torch.Tensor, down: torch.Tensor,
                          ds: torch.Tensor) -> torch.Tensor:
    tokens, hidden = x.shape
    top = ids.shape[1]
    mid = down.shape[-1]
    group = hidden // gs.shape[-1]
    block_k, warps = LAUNCH_CONFIGS.get(tokens, (256, 4))
    intermediate = torch.empty((tokens * top, mid), device=x.device, dtype=x.dtype)
    outputs = torch.empty((tokens * top, hidden), device=x.device, dtype=x.dtype)
    wrap_triton(_project)[(tokens * top, mid)](x, gu, gs, ids, intermediate,
        2 * mid, hidden, top, group, True, min(block_k, hidden), num_warps=warps, enable_fp_fusion=False)
    wrap_triton(_project)[(tokens * top, hidden)](intermediate, down, ds, ids, outputs,
        hidden, mid, top, group, False, min(block_k, mid), num_warps=warps, enable_fp_fusion=False)
    result = torch.empty_like(x)
    wrap_triton(_combine)[(tokens, triton.cdiv(hidden, 256))](
        outputs, weights, result, hidden, top, 256, enable_fp_fusion=False)
    return result


@selected_int8_experts.register_fake
def _fake_selected_int8_experts(x, ids, weights, gu, gs, down, ds):
    return torch.empty_like(x)
