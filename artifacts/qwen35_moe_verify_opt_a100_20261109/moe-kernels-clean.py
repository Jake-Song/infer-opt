"""Selected-expert W8A16 kernels for small decode/verification batches."""
import torch
import triton
import triton.language as tl
from torch.library import custom_op, wrap_triton

# Explicit tuning overrides keyed by activation rows. Defaults retain validated arithmetic.
LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}
# Explicit verification override: None selects the reference kernel.
PACKED_PAIRS_CONFIGS: dict[int, tuple[int, int] | None] = {}
USE_PACKED_VERIFY = False


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


@triton.jit
def _project_packed(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr, TOP: tl.constexpr, GROUP: tl.constexpr, GATE: tl.constexpr):
    """Load adjacent BF16/FP16 and INT8 pairs; preserve the reference K=256 accumulation."""
    route = tl.program_id(0)
    row = tl.program_id(1)
    expert = tl.load(IDS + route).to(tl.int64)
    token = route // TOP
    col = tl.arange(0, 128)
    even = tl.full((128,), 0, tl.float32)
    odd = tl.full((128,), 0, tl.float32)
    up_even = tl.full((128,), 0, tl.float32)
    up_odd = tl.full((128,), 0, tl.float32)
    for start in range(tl.cdiv(K, 256)):
        k = start * 256 + 2 * col
        base_x = (token if GATE else route) * K
        base_q = expert * N * K + row * K
        xb = tl.load(X.to(tl.pointer_type(tl.uint32)) + (base_x + k) // 2, k < K, 0)
        qb = tl.load(Q.to(tl.pointer_type(tl.uint16)) + (base_q + k) // 2, k < K, 0)
        x0 = xb.to(tl.uint16).to(X.dtype.element_ty, bitcast=True).to(tl.float32)
        x1 = (xb >> 16).to(tl.uint16).to(X.dtype.element_ty, bitcast=True).to(tl.float32)
        q0 = qb.to(tl.int8).to(tl.float32)
        q1 = (qb >> 8).to(tl.int8).to(tl.float32)
        base_s = (expert * N + row) * (K // GROUP)
        s0 = tl.load(S + base_s + k // GROUP, k < K, 0).to(tl.float32)
        s1 = s0
        even += x0 * q0 * s0
        odd += x1 * q1 * s1
        if GATE:
            qb2 = tl.load(Q.to(tl.pointer_type(tl.uint16)) + (base_q + N // 2 * K + k) // 2, k < K, 0)
            q2 = qb2.to(tl.int8).to(tl.float32)
            q3 = (qb2 >> 8).to(tl.int8).to(tl.float32)
            s2 = tl.load(S + base_s + N // 2 * (K // GROUP) + k // GROUP, k < K, 0).to(tl.float32)
            s3 = s2
            up_even += x0 * q2 * s2
            up_odd += x1 * q3 * s3
    groups = tl.sum(tl.reshape(even + odd, (4, 32)), 1)
    group_index = tl.arange(0, 4)
    value = tl.sum(tl.where(group_index % 2 == 0, groups, 0), 0) + tl.sum(tl.where(group_index % 2 == 1, groups, 0), 0)
    if GATE:
        groups_up = tl.sum(tl.reshape(up_even + up_odd, (4, 32)), 1)
        up = tl.sum(tl.where(group_index % 2 == 0, groups_up, 0), 0) + tl.sum(tl.where(group_index % 2 == 1, groups_up, 0), 0)
        value = value.to(X.dtype.element_ty).to(tl.float32)
        up = up.to(X.dtype.element_ty).to(tl.float32)
        value = (value * tl.sigmoid(value)).to(X.dtype.element_ty).to(tl.float32) * up
    tl.store(Y + route * (N // 2 if GATE else N) + row, value)


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
    default_packed = (USE_PACKED_VERIFY and tokens in (2, 4) and hidden == 2048
                      and mid == 512 and top == 8 and group == 128
                      and x.dtype == torch.bfloat16 and tokens not in LAUNCH_CONFIGS
                      and x.is_cuda and torch.cuda.get_device_capability(x.device) == (8, 0))
    packed = PACKED_PAIRS_CONFIGS.get(tokens, (256, 1) if default_packed else None)
    if packed is not None:
        if packed[0] != 256 or hidden % 256 or mid % 256 or group % 2:
            raise ValueError('packed expert kernel requires K blocks of 256 and even scale groups')
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError('packed expert kernel requires BF16 or FP16 activations')
        packed_warps = packed[1]
        wrap_triton(_project_packed)[(tokens * top, mid)](x, gu, gs, ids, intermediate,
            2 * mid, hidden, top, group, True, num_warps=packed_warps, enable_fp_fusion=False)
        wrap_triton(_project_packed)[(tokens * top, hidden)](intermediate, down, ds, ids, outputs,
            hidden, mid, top, group, False, num_warps=packed_warps, enable_fp_fusion=False)
    else:
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
