"""Experimental route-grouped W8A16 prefill; decode dispatch stays in moe_kernels.

Routes and tile counts stay on the GPU. No expert-bank BF16 copy is retained.
Group scales multiply FP32 partial sums, so weights are never rounded to BF16;
outputs therefore differ from the dequantize + ``F.linear`` loop by rounding.
Backends are judged by distance to :func:`float64_int8_experts`, see
:func:`float64_gate`.
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
            # BK divides GROUP, so one scale per output column covers the block.
            s = tl.load(S + expert * N * (K // GROUP) + cols * (K // GROUP)
                        + (block * BK) // GROUP, cols < N, other=0)
            # INT8 values are exact in BF16/FP16; the scale is applied in FP32,
            # as in the small-row GEMV, instead of rounding q * scale to BF16.
            acc += tl.dot(a, q.to(a.dtype)) * s.to(tl.float32)[None, :]
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


def route_schedule(ids, experts):
    """Stable expert-sorted assignments, per-expert row counts and cumulative row ends."""
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(experts, device=ids.device, dtype=torch.int32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return order, counts, counts.cumsum(0, dtype=torch.int32)


def tile_schedule(counts, block_m, rows=None):
    """Cumulative row-tile ends per expert.

    ``rows=(low, high)`` schedules only experts with ``low < count <= high``; the
    kernel's tile search skips the others, so two tile shapes can share one route sort.
    """
    if rows is not None:
        counts = torch.where((counts > rows[0]) & (counts <= rows[1]), counts, torch.zeros_like(counts))
    return ((counts + block_m - 1) // block_m).cumsum(0, dtype=torch.int32)


def _check_tile(block_m, block_n, block_k, group):
    if block_m not in (16, 32, 64) or block_n not in (32, 64, 128) or block_k not in (16, 32, 64, 128):
        raise ValueError('unsupported GEMM tile')
    if group % block_k:
        raise ValueError('block_k must divide the scale group')


def grouped_int8_experts(x, ids, weights, gu, gs, down, ds, *, block_m=16, block_n=64, block_k=None,
                         num_warps=4, large=None, stages=None):
    """Evaluate unique top-k routes, with BF16/FP16 group-scaled INT8 weights.

    Explicit experimental entry point. Callers must pass contiguous CUDA tensors;
    routing IDs follow torch.topk's unique, in-range expert contract.

    ``large=(rows, block_m, block_n, block_k, num_warps)`` gives experts with more
    than ``rows`` assignments their own tile shape: a wide row tile reads a busy
    expert's weights fewer times, while a narrow one wastes less on a few rows.
    Each expert is still evaluated by exactly one launch per projection.

    A ``stages`` dictionary receives the intermediate tensors for numerical
    diagnosis: ``gate_up``/``activated`` in expert-sorted row order, ``routed``
    by token/slot.
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
    if block_k is None:
        block_k = min(64, group)
    _check_tile(block_m, block_n, block_k, group)
    t, top = ids.shape
    if not 0 < top <= e:
        raise ValueError('invalid top-k')
    # (row range, tile) classes; every expert falls in exactly one.
    classes = [(None, (block_m, block_n, block_k, num_warps))]
    if large is not None:
        rows, *tile = large
        if rows < 0 or len(tile) != 4:
            raise ValueError('large must be (rows, block_m, block_n, block_k, num_warps)')
        _check_tile(*tile[:3], group)
        classes = [((0, rows), classes[0][1]), ((rows, t), tuple(tile))]
    if t == 0:
        return torch.empty_like(x)
    order, counts, ends = route_schedule(ids, e)
    gate_up = torch.empty((t * top, n), device=x.device, dtype=x.dtype)
    activated = torch.empty((t * top, mid), device=x.device, dtype=x.dtype)
    routed = torch.empty((t * top, h), device=x.device, dtype=x.dtype)
    schedules = []
    for rows, (bm, bn, bk, warps) in classes:
        # Sum ceil(count_e / BM) <= ceil(total / BM) + E - 1: a grid bound without a host read.
        schedules.append((tile_schedule(counts, bm, rows), triton.cdiv(t * top, bm) + e - 1, bm, bn, bk, warps))
    for tiles, max_tiles, bm, bn, bk, warps in schedules:
        _grouped[(max_tiles, triton.cdiv(n, bn))](
            x, gu, gs, order, ends, tiles, gate_up, e, h, n, top, group, True, bm, bn, bk, num_warps=warps)
    _swiglu[(triton.cdiv(t * top * mid, 256),)](
        gate_up, activated, t * top * mid, mid, 256, enable_fp_fusion=False)
    for tiles, max_tiles, bm, bn, bk, warps in schedules:
        _grouped[(max_tiles, triton.cdiv(h, bn))](
            activated, down, ds, order, ends, tiles, routed, e, mid, h, top, group, False, bm, bn, bk,
            num_warps=warps)
    result = torch.empty_like(x)
    _combine[(t, triton.cdiv(h, 256))](routed, ids, weights, result, h, top,
        triton.next_power_of_2(top), 256, enable_fp_fusion=False)
    if stages is not None:
        stages.update(order=order, gate_up=gate_up, activated=activated, routed=routed)
    return result


def w8a16_gemm(x, weight, scale, *, block_m=64, block_n=64, block_k=None, num_warps=4):
    """``x @ (q * scale)^T`` for one group-scaled INT8 matrix, without a BF16 copy of it.

    The routed-expert kernel with a single expert: scales multiply FP32 partial
    sums. For comparing dense prefill projections with dequantize + ``F.linear``.
    """
    if x.ndim != 2 or weight.ndim != 2 or scale.ndim != 2 or weight.shape[1] != x.shape[1]:
        raise ValueError('expected x[T,K], weight[N,K] and scale[N,groups]')
    if not all(t.is_cuda and t.is_contiguous() for t in (x, weight, scale)):
        raise ValueError('expected contiguous CUDA tensors')
    if x.dtype not in (torch.bfloat16, torch.float16) or scale.dtype != x.dtype or weight.dtype != torch.int8:
        raise ValueError('expected BF16/FP16 activations and scales with an INT8 weight')
    t, k = x.shape
    n = weight.shape[0]
    group = k // scale.shape[1]
    if scale.shape[0] != n or group * scale.shape[1] != k:
        raise ValueError('incompatible group scales')
    if block_k is None:
        block_k = min(64, group)
    _check_tile(block_m, block_n, block_k, group)
    y = torch.empty((t, n), device=x.device, dtype=x.dtype)
    if t:
        order = torch.arange(t, device=x.device)
        ends = torch.tensor([t], device=x.device, dtype=torch.int32)
        tiles = torch.tensor([triton.cdiv(t, block_m)], device=x.device, dtype=torch.int32)
        _grouped[(triton.cdiv(t, block_m), triton.cdiv(n, block_n))](
            x, weight, scale, order, ends, tiles, y, 1, k, n, 1, group, True, block_m, block_n, block_k,
            num_warps=num_warps)
    return y


@torch.no_grad()
def float64_int8_experts(x, ids, weights, gu, gs, down, ds, *, stages=None):
    """Unrounded FP64 evaluation of the routed experts; weights are exactly ``q * scale``.

    This is the accuracy yardstick for prefill backends: every BF16 path rounds
    somewhere (dequantized weights, projections, SwiGLU, accumulation), so paths
    are ranked by their distance to this value instead of to each other.
    ``stages`` receives FP64 intermediates in the layout of ``grouped_int8_experts``.
    """
    e, n, h = gu.shape
    top = ids.shape[1]
    flat = ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=e).tolist()
    flat_weights = weights.reshape(-1).double()
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float64)
    if stages is not None:
        rows = flat.numel()
        stages.update(order=order,
                      gate_up=torch.empty((rows, n), device=x.device, dtype=torch.float64),
                      activated=torch.empty((rows, n // 2), device=x.device, dtype=torch.float64),
                      routed=torch.empty((rows, h), device=x.device, dtype=torch.float64))
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        selected = order[start:start + count]
        tokens = selected // top
        w1 = gu[expert].double() * gs[expert].double().repeat_interleave(h // gs.shape[-1], -1)
        w2 = down[expert].double() * ds[expert].double().repeat_interleave(n // 2 // ds.shape[-1], -1)
        gate_up = x[tokens].double() @ w1.T
        gate, up = gate_up.chunk(2, -1)
        activated = torch.nn.functional.silu(gate) * up
        y = activated @ w2.T
        out.index_add_(0, tokens, y * flat_weights[selected, None])
        if stages is not None:
            stages['gate_up'][start:start + count] = gate_up
            stages['activated'][start:start + count] = activated
            stages['routed'][selected] = y
        start += count
    return out


#: Allowance for rounding ties when two backends are statistically as accurate.
FLOAT64_GATE_SLACK = .01


def float64_gate(candidate, existing, truth, slack=FLOAT64_GATE_SLACK):
    """Accept a backend whose RMS error to the FP64 value does not exceed the adopted path's.

    Element-wise closeness to another BF16 path is not a usable criterion: each
    path is about half a percent from the FP64 value through its own roundings.
    """
    def error(got):
        delta = got.double() - truth
        return {'rms_relative': (delta.square().mean().sqrt() / truth.square().mean().sqrt()).item(),
                'max_abs': delta.abs().max().item()}
    got, base = error(candidate), error(existing)
    return {'candidate': got, 'existing': base, 'slack': slack,
            'passes': got['rms_relative'] <= base['rms_relative'] * (1 + slack)}
