"""Experimental route-grouped W8A16 prefill; decode dispatch stays in moe_kernels.

Routes and tile counts stay on the GPU. No expert-bank BF16 copy is retained.
Tensor-core reduction order can differ from the eager expert reference.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped(X, Q, S, Order, Ends, Tiles, Y,
             E: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             TOP: tl.constexpr, GROUP: tl.constexpr, FIRST: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    if tile < tl.load(Tiles + E - 1):
        # Upper-bound search skips experts with zero assignments.
        lo, hi = 0, E
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(Tiles + mid)
            lo = tl.where(tile >= end, mid + 1, lo)
            hi = tl.where(tile >= end, hi, mid)
        expert = lo
        prior_tiles = tl.load(Tiles + expert - 1, expert > 0, other=0)
        start = tl.load(Ends + expert - 1, expert > 0, other=0)
        end = tl.load(Ends + expert)
        rows = start + (tile - prior_tiles) * BM + tl.arange(0, BM)
        assignment = tl.load(Order + rows, rows < end, other=0)
        source = assignment // TOP if FIRST else rows
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + kk
            a = tl.load(X + source[:, None] * K + k[None, :],
                        (rows[:, None] < end) & (k[None, :] < K), other=0)
            q = tl.load(Q + expert * N * K + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
            s = tl.load(S + expert * N * (K // GROUP) + cols[None, :] * (K // GROUP)
                        + k[:, None] // GROUP,
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
            b = (q.to(a.dtype) * s.to(a.dtype)).to(a.dtype)
            acc = tl.dot(a, b, acc)
        dest = rows if FIRST else assignment
        tl.store(Y + dest[:, None] * N + cols[None, :], acc.to(Y.dtype.element_ty),
                 (rows[:, None] < end) & (cols[None, :] < N))


@triton.jit
def _swiglu(GU, Y, SIZE: tl.constexpr, MID: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = p // MID, p % MID
    gate = tl.load(GU + row * 2 * MID + col, p < SIZE, other=0).to(tl.float32)
    up = tl.load(GU + row * 2 * MID + MID + col, p < SIZE, other=0)
    silu = (gate * tl.sigmoid(gate)).to(up.dtype)
    tl.store(Y + p, (silu.to(tl.float32) * up.to(tl.float32)).to(up.dtype), p < SIZE)


@triton.jit
def _combine(Y, Ids, W, Out, H: tl.constexpr, TOP: tl.constexpr,
             BT: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, BT)
    ids = tl.load(Ids + token * TOP + slots, slots < TOP, other=2147483647).to(tl.int32)
    prev = -1
    acc = tl.full((BLOCK,), 0, tl.float32)
    for _ in range(TOP):
        expert = tl.min(tl.where(ids > prev, ids, 2147483647), 0)
        slot = tl.min(tl.where(ids == expert, slots, 2147483647), 0)
        y = tl.load(Y + (token * TOP + slot) * H + col, col < H, other=0)
        weight = tl.load(W + token * TOP + slot)
        contribution = (y.to(tl.float32) * weight.to(tl.float32)).to(y.dtype)
        acc = (acc + contribution.to(tl.float32)).to(y.dtype).to(tl.float32)
        prev = expert
    tl.store(Out + token * H + col, acc, col < H)


def grouped_int8_experts(x, ids, weights, gu, gs, down, ds, *, block_m=16, block_n=64, block_k=64):
    """Evaluate unique top-k routes, with BF16/FP16 group-scaled INT8 weights.

    Explicit experimental entry point. Callers must pass contiguous CUDA tensors;
    routing IDs follow torch.topk's unique, in-range expert contract.
    """
    tensors = (x, ids, weights, gu, gs, down, ds)
    if not all(t.is_cuda and t.device == x.device and t.is_contiguous() for t in tensors):
        raise ValueError('grouped experts require contiguous tensors on one CUDA device')
    if x.ndim != 2 or ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != x.shape[0]:
        raise ValueError('expected x[T,H] and routes/weights[T,top_k]')
    if x.dtype not in (torch.bfloat16, torch.float16) or any(t.dtype != x.dtype for t in (weights, gs, ds)):
        raise ValueError('activations, scales and routing weights must share BF16/FP16 dtype')
    if gu.dtype != torch.int8 or down.dtype != torch.int8 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('expected INT8 banks and integer route IDs')
    if gu.ndim != 3 or down.ndim != 3:
        raise ValueError('expected 3-D expert banks')
    e, n, h = gu.shape
    mid = n // 2
    if h != x.shape[1] or n % 2 or down.shape != (e, h, mid) or gs.ndim != 3 or gs.shape[:2] != (e, n):
        raise ValueError('incompatible expert shapes')
    if not gs.shape[-1] or not h or not mid or not e:
        raise ValueError('expert dimensions and scale groups must be positive')
    group = h // gs.shape[-1]
    if group <= 0 or h % group or mid % group or gs.shape[-1] * group != h or ds.shape != (e, h, mid // group):
        raise ValueError('incompatible group scales')
    if block_m not in (16, 32, 64) or block_n not in (32, 64, 128) or block_k not in (32, 64, 128):
        raise ValueError('unsupported GEMM tile')
    t, top = ids.shape
    if not 0 < top <= e:
        raise ValueError('invalid top-k')
    if t == 0:
        return torch.empty_like(x)
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(e, device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    ends = counts.cumsum(0, dtype=torch.int32)
    tiles = ((counts + block_m - 1) // block_m).cumsum(0, dtype=torch.int32)
    # Sum ceil(count_e / BM) <= ceil(total / BM) + E - 1.
    max_tiles = triton.cdiv(t * top, block_m) + e - 1
    gate_up = torch.empty((t * top, n), device=x.device, dtype=x.dtype)
    activated = torch.empty((t * top, mid), device=x.device, dtype=x.dtype)
    routed = torch.empty((t * top, h), device=x.device, dtype=x.dtype)
    _grouped[(max_tiles, triton.cdiv(n, block_n))](
        x, gu, gs, order, ends, tiles, gate_up, e, h, n, top, group, True,
        block_m, block_n, block_k, num_warps=4)
    _swiglu[(triton.cdiv(t * top * mid, 256),)](
        gate_up, activated, t * top * mid, mid, 256, enable_fp_fusion=False)
    _grouped[(max_tiles, triton.cdiv(h, block_n))](
        activated, down, ds, order, ends, tiles, routed, e, mid, h, top, group, False,
        block_m, block_n, block_k, num_warps=4)
    result = torch.empty_like(x)
    _combine[(t, triton.cdiv(h, 256))](routed, ids, weights, result, h, top,
        triton.next_power_of_2(top), 256, enable_fp_fusion=False)
    return result
