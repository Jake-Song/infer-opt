"""Selected-expert W8A16 kernels for small decode/verification batches."""
import torch
import triton
import triton.language as tl
from torch.library import custom_op, wrap_triton

# Explicit tuning overrides keyed by activation rows. Defaults retain validated arithmetic.
LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}
# Experimental verification kernel. Keep selection explicit until full-model validation.
SHARED_ROUTE_CONFIGS: dict[int, tuple[int, int, int]] = {}
TILED_ROUTE_CONFIGS: dict[int, tuple[int, int, int]] = {}
GROUPED_DOT_CONFIGS: dict[int, tuple[int, int, int]] = {}
INT32_LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}
ORDERED_LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}
PAIRED_LAUNCH_CONFIGS: dict[int, tuple[int, int]] = {}
PACKED_PAIRS_CONFIGS: dict[int, tuple[int, int]] = {}


@triton.jit
def _project(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr, TOP: tl.constexpr,
             GROUP: tl.constexpr, GATE: tl.constexpr, BK: tl.constexpr,
             INDEX32: tl.constexpr = False, ORDERED: tl.constexpr = False):
    route = tl.program_id(0)
    row = tl.program_id(1)
    expert = tl.load(IDS + route).to(tl.int32 if INDEX32 else tl.int64)
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
    if ORDERED and BK == 256:
        # Original 4-warp reduction: two values/thread, 32 lanes, four warps.
        value = tl.sum(tl.sum(tl.sum(tl.reshape(acc, (2, 4, 32)), 0), 1), 0)
    else:
        value = tl.sum(acc, 0)
    if GATE:
        # Match the projection's BF16 rounding before SwiGLU.
        value = value.to(X.dtype.element_ty).to(tl.float32)
        if ORDERED and BK == 256:
            up = tl.sum(tl.sum(tl.sum(tl.reshape(other, (2, 4, 32)), 0), 1), 0)
        else:
            up = tl.sum(other, 0)
        up = up.to(X.dtype.element_ty).to(tl.float32)
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
def _project_pairs(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr,
                   TOP: tl.constexpr, GROUP: tl.constexpr, GATE: tl.constexpr,
                   PACKED: tl.constexpr = False):
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
        if PACKED:
            xb = tl.load(X.to(tl.pointer_type(tl.uint32)) + (base_x + k) // 2, k < K, 0)
            qb = tl.load(Q.to(tl.pointer_type(tl.uint16)) + (base_q + k) // 2, k < K, 0)
            x0 = xb.to(tl.uint16).to(X.dtype.element_ty, bitcast=True).to(tl.float32)
            x1 = (xb >> 16).to(tl.uint16).to(X.dtype.element_ty, bitcast=True).to(tl.float32)
            q0 = qb.to(tl.int8).to(tl.float32)
            q1 = (qb >> 8).to(tl.int8).to(tl.float32)
        else:
            x0 = tl.load(X + base_x + k, k < K, 0).to(tl.float32)
            x1 = tl.load(X + base_x + k + 1, k + 1 < K, 0).to(tl.float32)
            q0 = tl.load(Q + base_q + k, k < K, 0).to(tl.float32)
            q1 = tl.load(Q + base_q + k + 1, k + 1 < K, 0).to(tl.float32)
        base_s = (expert * N + row) * (K // GROUP)
        s0 = tl.load(S + base_s + k // GROUP, k < K, 0).to(tl.float32)
        if PACKED:
            s1 = s0
        else:
            s1 = tl.load(S + base_s + (k + 1) // GROUP, k + 1 < K, 0).to(tl.float32)
        even += x0 * q0 * s0
        odd += x1 * q1 * s1
        if GATE:
            if PACKED:
                qb2 = tl.load(Q.to(tl.pointer_type(tl.uint16)) + (base_q + (N // 2) * K + k) // 2, k < K, 0)
                q2 = qb2.to(tl.int8).to(tl.float32)
                q3 = (qb2 >> 8).to(tl.int8).to(tl.float32)
            else:
                q2 = tl.load(Q + base_q + (N // 2) * K + k, k < K, 0).to(tl.float32)
                q3 = tl.load(Q + base_q + (N // 2) * K + k + 1, k + 1 < K, 0).to(tl.float32)
            s2 = tl.load(S + base_s + (N // 2) * (K // GROUP) + k // GROUP, k < K, 0).to(tl.float32)
            if PACKED:
                s3 = s2
            else:
                s3 = tl.load(S + base_s + (N // 2) * (K // GROUP) + (k + 1) // GROUP, k + 1 < K, 0).to(tl.float32)
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


@triton.jit
def _project_tiled(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr,
                   TOP: tl.constexpr, GROUP: tl.constexpr, GATE: tl.constexpr,
                   BK: tl.constexpr, BN: tl.constexpr):
    route = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(IDS + route).to(tl.int64)
    token = route // TOP
    width: tl.constexpr = N // 2 if GATE else N
    col = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    other = tl.full((BN, BK), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + col
        x = tl.load(X + (token if GATE else route) * K + k, k < K, 0).to(tl.float32)
        offsets = expert * N * K + rows[:, None] * K + k[None, :]
        scales = (expert * N + rows[:, None]) * (K // GROUP) + k[None, :] // GROUP
        mask = (rows[:, None] < width) & (k[None, :] < K)
        q = tl.load(Q + offsets, mask, 0).to(tl.float32)
        s = tl.load(S + scales, mask, 0).to(tl.float32)
        acc += x[None, :] * q * s
        if GATE:
            q2 = tl.load(Q + offsets + (N // 2) * K, mask, 0).to(tl.float32)
            s2 = tl.load(S + scales + (N // 2) * (K // GROUP), mask, 0).to(tl.float32)
            other += x[None, :] * q2 * s2
    value = tl.sum(acc, 1)
    if GATE:
        value = value.to(X.dtype.element_ty).to(tl.float32)
        up = tl.sum(other, 1).to(X.dtype.element_ty).to(tl.float32)
        value = (value * tl.sigmoid(value)).to(X.dtype.element_ty).to(tl.float32) * up
    tl.store(Y + route * width + rows, value, rows < width)


@triton.jit
def _project_grouped_dot(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr,
                         TOP: tl.constexpr, TOKENS: tl.constexpr, GROUP: tl.constexpr,
                         GATE: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                         BR: tl.constexpr, BS: tl.constexpr):
    """Grouped Tensor Core candidate; BF16/FP16 weight rounding differs from GEMV."""
    anchor = tl.program_id(0)
    expert = tl.load(IDS + anchor).to(tl.int64)
    routes = tl.arange(0, BR)
    all_ids = tl.load(IDS + routes, routes < TOKENS * TOP, -1)
    earlier = tl.sum(((routes < anchor) & (all_ids == expert)).to(tl.int32), 0)
    if earlier == 0:
        tokens = tl.arange(0, 16)
        slots = tl.arange(0, BS)
        token_ids = tl.load(IDS + tokens[:, None] * TOP + slots[None, :],
                           (tokens[:, None] < TOKENS) & (slots[None, :] < TOP), -1)
        slot = tl.max(tl.where(token_ids == expert, slots[None, :], -1), 1)
        active = (tokens < TOKENS) & (slot >= 0)
        route = tokens * TOP + slot
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        width: tl.constexpr = N // 2 if GATE else N
        col = tl.arange(0, BK)
        acc = tl.full((16, BN), 0, tl.float32)
        other = tl.full((16, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + col
            xrow = tokens if GATE else route
            x = tl.load(X + xrow[:, None] * K + k[None, :],
                        active[:, None] & (k[None, :] < K), 0)
            offset = expert * N * K + rows[:, None] * K + k[None, :]
            scale_offset = (expert * N + rows[:, None]) * (K // GROUP) + k[None, :] // GROUP
            mask = (rows[:, None] < width) & (k[None, :] < K)
            q = tl.load(Q + offset, mask, 0).to(tl.float32)
            s = tl.load(S + scale_offset, mask, 0).to(tl.float32)
            weight = (q * s).to(X.dtype.element_ty)
            acc = tl.dot(x, tl.trans(weight), acc)
            if GATE:
                q2 = tl.load(Q + offset + (N // 2) * K, mask, 0).to(tl.float32)
                s2 = tl.load(S + scale_offset + (N // 2) * (K // GROUP), mask, 0).to(tl.float32)
                weight2 = (q2 * s2).to(X.dtype.element_ty)
                other = tl.dot(x, tl.trans(weight2), other)
        value = acc
        if GATE:
            value = value.to(X.dtype.element_ty).to(tl.float32)
            up = other.to(X.dtype.element_ty).to(tl.float32)
            value = (value * tl.sigmoid(value)).to(X.dtype.element_ty).to(tl.float32) * up
        tl.store(Y + route[:, None] * width + rows[None, :], value,
                 active[:, None] & (rows[None, :] < width))


@triton.jit
def _project_shared_routes(X, Q, S, IDS, Y, N: tl.constexpr, K: tl.constexpr,
                           TOP: tl.constexpr, TOKENS: tl.constexpr, GROUP: tl.constexpr,
                           GATE: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                           BT: tl.constexpr, BR: tl.constexpr, BS: tl.constexpr):
    """The first occurrence of an expert computes every token routed to it.

    IDs remain dynamic graph inputs. Routing top-k supplies unique experts within
    each token. Preserve the original BK accumulation, projection rounding and
    route order; only weight loads are shared across token rows.
    """
    anchor = tl.program_id(0)
    expert = tl.load(IDS + anchor).to(tl.int64)
    routes = tl.arange(0, BR)
    all_ids = tl.load(IDS + routes, routes < TOKENS * TOP, -1)
    earlier = tl.sum(((routes < anchor) & (all_ids == expert)).to(tl.int32), 0)
    if earlier == 0:
        tokens = tl.arange(0, BT)
        slots = tl.arange(0, BS)
        token_ids = tl.load(IDS + tokens[:, None] * TOP + slots[None, :],
                            (tokens[:, None] < TOKENS) & (slots[None, :] < TOP), -1)
        slot = tl.max(tl.where(token_ids == expert, slots[None, :], -1), 1)
        active = (tokens < TOKENS) & (slot >= 0)
        route = tokens * TOP + slot
        rows = tl.program_id(1) * BN + tl.arange(0, BN)
        width: tl.constexpr = N // 2 if GATE else N
        col = tl.arange(0, BK)
        acc = tl.full((BT, BN, BK), 0, tl.float32)
        other = tl.full((BT, BN, BK), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + col
            xrow = tokens if GATE else route
            x = tl.load(X + xrow[:, None] * K + k[None, :],
                        active[:, None] & (k[None, :] < K), 0).to(tl.float32)
            offset = expert * N * K + rows[:, None] * K + k[None, :]
            scale_offset = (expert * N + rows[:, None]) * (K // GROUP) + k[None, :] // GROUP
            mask = (rows[:, None] < width) & (k[None, :] < K)
            q = tl.load(Q + offset, mask, 0).to(tl.float32)
            s = tl.load(S + scale_offset, mask, 0).to(tl.float32)
            acc += x[:, None, :] * q[None, :, :] * s[None, :, :]
            if GATE:
                q2 = tl.load(Q + offset + (N // 2) * K, mask, 0).to(tl.float32)
                s2 = tl.load(S + scale_offset + (N // 2) * (K // GROUP), mask, 0).to(tl.float32)
                other += x[:, None, :] * q2[None, :, :] * s2[None, :, :]
        value = tl.sum(acc, 2)
        if GATE:
            value = value.to(X.dtype.element_ty).to(tl.float32)
            up = tl.sum(other, 2).to(X.dtype.element_ty).to(tl.float32)
            value = (value * tl.sigmoid(value)).to(X.dtype.element_ty).to(tl.float32) * up
        tl.store(Y + route[:, None] * width + rows[None, :], value,
                 active[:, None] & (rows[None, :] < width))


@custom_op('infer_opt::selected_int8_experts', mutates_args=())
def selected_int8_experts(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                          gu: torch.Tensor, gs: torch.Tensor, down: torch.Tensor,
                          ds: torch.Tensor) -> torch.Tensor:
    tokens, hidden = x.shape
    top = ids.shape[1]
    mid = down.shape[-1]
    group = hidden // gs.shape[-1]
    block_k, warps = LAUNCH_CONFIGS.get(tokens, (256, 4))
    index32_config = INT32_LAUNCH_CONFIGS.get(tokens)
    ordered_config = ORDERED_LAUNCH_CONFIGS.get(tokens)
    index32 = False
    if index32_config is not None:
        block_k, warps = index32_config
        # Offset arithmetic must fit for both quantized weights and their scales.
        index32 = gu.numel() < 2**31 and down.numel() < 2**31
    if ordered_config is not None:
        block_k, warps = ordered_config
    intermediate = torch.empty((tokens * top, mid), device=x.device, dtype=x.dtype)
    outputs = torch.empty((tokens * top, hidden), device=x.device, dtype=x.dtype)
    shared = SHARED_ROUTE_CONFIGS.get(tokens)
    tiled = TILED_ROUTE_CONFIGS.get(tokens)
    grouped = GROUPED_DOT_CONFIGS.get(tokens)
    paired = PAIRED_LAUNCH_CONFIGS.get(tokens)
    packed = PACKED_PAIRS_CONFIGS.get(tokens)
    if paired is not None or packed is not None:
        _, warps = packed if packed is not None else paired
        wrap_triton(_project_pairs)[(tokens * top, mid)](x, gu, gs, ids, intermediate,
            2 * mid, hidden, top, group, True, packed is not None, num_warps=warps, enable_fp_fusion=False)
        wrap_triton(_project_pairs)[(tokens * top, hidden)](intermediate, down, ds, ids, outputs,
            hidden, mid, top, group, False, packed is not None, num_warps=warps, enable_fp_fusion=False)
    elif grouped is not None:
        block_k, block_n, warps = grouped
        wrap_triton(_project_grouped_dot)[(tokens * top, triton.cdiv(mid, block_n))](
            x, gu, gs, ids, intermediate, 2 * mid, hidden, top, tokens, group, True,
            block_k, block_n, triton.next_power_of_2(tokens * top),
            triton.next_power_of_2(top), num_warps=warps, enable_fp_fusion=False)
        wrap_triton(_project_grouped_dot)[(tokens * top, triton.cdiv(hidden, block_n))](
            intermediate, down, ds, ids, outputs, hidden, mid, top, tokens, group, False,
            block_k, block_n, triton.next_power_of_2(tokens * top),
            triton.next_power_of_2(top), num_warps=warps, enable_fp_fusion=False)
    elif tiled is not None:
        block_k, block_n, warps = tiled
        wrap_triton(_project_tiled)[(tokens * top, triton.cdiv(mid, block_n))](
            x, gu, gs, ids, intermediate, 2 * mid, hidden, top, group, True,
            min(block_k, hidden), block_n, num_warps=warps, enable_fp_fusion=False)
        wrap_triton(_project_tiled)[(tokens * top, triton.cdiv(hidden, block_n))](
            intermediate, down, ds, ids, outputs, hidden, mid, top, group, False,
            min(block_k, mid), block_n, num_warps=warps, enable_fp_fusion=False)
    elif shared is None:
        wrap_triton(_project)[(tokens * top, mid)](x, gu, gs, ids, intermediate,
            2 * mid, hidden, top, group, True, min(block_k, hidden), index32, ordered_config is not None, num_warps=warps, enable_fp_fusion=False)
        wrap_triton(_project)[(tokens * top, hidden)](intermediate, down, ds, ids, outputs,
            hidden, mid, top, group, False, min(block_k, mid), index32, ordered_config is not None, num_warps=warps, enable_fp_fusion=False)
    else:
        block_k, block_n, warps = shared
        common = dict(num_warps=warps, enable_fp_fusion=False)
        wrap_triton(_project_shared_routes)[(tokens * top, triton.cdiv(mid, block_n))](
            x, gu, gs, ids, intermediate, 2 * mid, hidden, top, tokens, group, True,
            min(block_k, hidden), block_n, triton.next_power_of_2(tokens),
            triton.next_power_of_2(tokens * top), triton.next_power_of_2(top), **common)
        wrap_triton(_project_shared_routes)[(tokens * top, triton.cdiv(hidden, block_n))](
            intermediate, down, ds, ids, outputs, hidden, mid, top, tokens, group, False,
            min(block_k, mid), block_n, triton.next_power_of_2(tokens),
            triton.next_power_of_2(tokens * top), triton.next_power_of_2(top), **common)
    result = torch.empty_like(x)
    wrap_triton(_combine)[(tokens, triton.cdiv(hidden, 256))](
        outputs, weights, result, hidden, top, 256, enable_fp_fusion=False)
    return result


@selected_int8_experts.register_fake
def _fake_selected_int8_experts(x, ids, weights, gu, gs, down, ds):
    return torch.empty_like(x)
