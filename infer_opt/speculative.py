"""Self-speculative greedy decoding for Qwen3.5 with its multi-token-prediction head (batch 1).

Batch-1 decode streams every weight once per token. Checking ``k`` drafted tokens and the one before them in one
``T = k + 1`` token forward streams them once too, so every accepted draft is a token for (almost) free. The drafts
come from the checkpoint's MTP head (:class:`~infer_opt.qwen3_5.Qwen3_5MTP`): one decoder layer on top of the
target's final hidden state, much cheaper than the target.

One round, all on the device with fixed shapes (one compiled function or one CUDA graph):

1. verify ``[y, d1..dk]`` at positions ``p..p+k``: target argmax ``g[0..k]`` and hidden ``h[p..p+k]``
2. accept ``n = 1 + (number of leading drafts with d[i] == g[i-1])``: the tokens ``g[0..n-1]`` are the greedy output
3. commit: the position moves by ``n`` and each GatedDeltaNet layer takes its state after token ``n - 1``
   (:meth:`~infer_opt.qwen3_5.Qwen3_5Cache.commit`); K/V past it are masked and later overwritten
4. the MTP head runs on the pairs ``(h[p+i], g[i])``, ``i = 0..k``: for every accepted ``i`` that is the true next
   token, so its KV for the accepted positions is right and its output at ``n - 1`` drafts ``d'1``
5. ``k - 1`` single-token MTP steps, each on its previous output, draft ``d'2..d'k``; the next input is
   ``[g[n-1], d'1..d'k]``, written into the input buffer so rounds chain without the host

The host reads ``n`` and ``g`` once per round. The output equals greedy decoding of the target up to floating-point
differences between a ``T``-token and a one-token forward (exact on the fp64 test model).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from .qwen3_5 import DECODE_MODES, Qwen3_5Cache, Qwen3_5ForCausalLM

__all__ = ["MTPSpeculativeDecoder", "SpecStats", "draft", "spec_round"]


def draft(model: Qwen3_5ForCausalLM, hidden: torch.Tensor, token: torch.Tensor, mtp_cache: Qwen3_5Cache,
          steps: int) -> list[torch.Tensor]:
    """``steps`` more drafts after ``token``, each from the MTP output (``hidden``) that drafted the previous one.
    Their MTP KV lands past ``mtp_cache.position``, which ends where it started."""
    drafts = []
    for _ in range(steps):
        hidden = model.mtp_hidden(token, hidden, mtp_cache, decode=True)
        token = model.lm_head(hidden).argmax(-1)
        mtp_cache.position.add_(1)
        drafts.append(token)
    if steps:
        mtp_cache.position.sub_(steps)
    return drafts


def spec_round(model: Qwen3_5ForCausalLM, tokens: torch.Tensor, cache: Qwen3_5Cache,
               mtp_cache: Qwen3_5Cache) -> tuple[torch.Tensor, torch.Tensor]:
    """One verify-commit-draft round on ``tokens`` (``[1, k + 1]``: the last target token and ``k`` drafts), which
    is overwritten with the next round's input. Returns the target's argmax ``[1, k + 1]`` and the number of its
    leading tokens that were accepted (``[1]``, 1..k+1)."""
    k = tokens.shape[1] - 1
    hidden = model.model.language_model.forward_verify(tokens, cache)
    target = model.lm_head(hidden).argmax(-1)
    accepted = (tokens[:, 1:] == target[:, :-1]).long().cumprod(-1).sum(-1) + 1
    cache.commit(accepted)
    mtp_out = model.mtp_hidden(target, hidden, mtp_cache, decode=True)
    mtp_cache.position.add_(accepted)
    last = accepted - 1
    hidden = mtp_out.index_select(1, last)
    first = model.lm_head(hidden).argmax(-1)
    drafts = draft(model, hidden, first, mtp_cache, k - 1)
    tokens.copy_(torch.cat([target.index_select(1, last), first, *drafts], dim=1))
    return target, accepted


@dataclass
class SpecStats:
    """Accepted tokens per round (1 = no draft accepted, k + 1 = all of them)."""

    accepted: Counter = field(default_factory=Counter)

    def record(self, n: int) -> None:
        self.accepted[n] += 1

    @property
    def rounds(self) -> int:
        return sum(self.accepted.values())

    @property
    def tokens_per_round(self) -> float:
        return sum(n * c for n, c in self.accepted.items()) / max(self.rounds, 1)

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "tokens_per_round": self.tokens_per_round,
                "accepted_histogram": {int(n): c for n, c in sorted(self.accepted.items())}}


class MTPSpeculativeDecoder:
    """Greedy decoding of one sequence with ``k`` MTP drafts per round (see the module docstring).

    ``mode`` is one of :data:`~infer_opt.qwen3_5.DECODE_MODES`, as for :class:`~infer_opt.qwen3_5.FixedShapeDecoder`:
    the round runs eagerly, through torch.compile (with CUDA graphs for ``compile-reduce-overhead``) or as one
    hand-captured CUDA graph (``cudagraph``). ``max_seq_len`` bounds prompt + generated tokens; the caches get room
    for one round's overshoot and the capture warm-up on top.
    """

    def __init__(self, model: Qwen3_5ForCausalLM, max_seq_len: int, k: int = 2, mode: str = "eager",
                 fullgraph: bool = True):
        if model.mtp is None:
            raise ValueError("the model has no MTP head: load it with from_pretrained_dir(..., mtp=True)")
        if mode not in DECODE_MODES:
            raise ValueError(f"mode must be one of {DECODE_MODES}, got {mode!r}")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.model, self.k, self.mode, self.max_seq_len = model, k, mode, max_seq_len
        self.tokens_per_round = k + 1
        room = max_seq_len + 4 * self.tokens_per_round + k  # a round's overshoot and three warm-up rounds
        self.cache = model.new_cache(1, room)
        self.cache.allocate_spec_buffers(self.tokens_per_round)
        self.mtp_cache = model.new_mtp_cache(room)
        self.tokens = torch.zeros(1, self.tokens_per_round, dtype=torch.long, device=self.cache.position.device)
        self.stats = SpecStats()
        self.graph = None
        self.fn = spec_round
        if mode.startswith("compile"):
            for t in self._state_tensors() + [self.tokens]:
                torch._dynamo.mark_static_address(t)
            options = {"mode": "reduce-overhead"} if mode == "compile-reduce-overhead" else {}
            self.fn = torch.compile(spec_round, fullgraph=fullgraph, dynamic=False, **options)

    def _state_tensors(self) -> list[torch.Tensor]:
        tensors = []
        for c in (self.cache, self.mtp_cache):
            groups = (c.keys, c.values, c.conv_states, c.recurrent_states, c.spec_conv, c.spec_states)
            tensors += [t for group in groups for t in group if t is not None] + [c.position]
        return tensors

    @torch.no_grad()
    def prefill(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Start a sequence: target prefill, MTP prefill on ``(h[j], x[j+1])`` and the first drafts. Returns the first
        generated token (``[1, 1]``)."""
        if input_ids.shape[0] != 1:
            raise ValueError("speculative decoding runs one sequence")
        model = self.model
        self.cache.reset()
        self.mtp_cache.reset()
        self.stats = SpecStats()
        hidden = model.model.language_model(input_ids, self.cache)
        first = model.lm_head(hidden[:, -1:]).argmax(-1)
        mtp_out = model.mtp_hidden(torch.cat([input_ids[:, 1:], first], dim=1), hidden, self.mtp_cache)[:, -1:]
        token = model.lm_head(mtp_out).argmax(-1)
        drafts = draft(model, mtp_out, token, self.mtp_cache, self.k - 1)
        self.tokens.copy_(torch.cat([first, token, *drafts], dim=1))
        return first

    @torch.no_grad()
    def prepare(self) -> None:
        """Capture the CUDA graph (``cudagraph`` mode). Can run at any time: the warm-up rounds scribble on the states,
        positions and input buffer, which are restored afterwards."""
        if self.mode != "cudagraph":
            return
        caches = (self.cache, self.mtp_cache)
        live = [t for c in caches for t in c.conv_states + c.recurrent_states + [c.position] if t is not None]
        live.append(self.tokens)
        saved = [t.clone() for t in live]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                spec_round(self.model, self.tokens, self.cache, self.mtp_cache)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = spec_round(self.model, self.tokens, self.cache, self.mtp_cache)
        for dst, src in zip(live, saved):
            dst.copy_(src)

    @torch.no_grad()
    def round(self) -> list[int]:
        """Run one round; returns the 1..k+1 tokens it generated."""
        cache = self.cache
        if cache.seen_tokens + self.tokens_per_round > cache.max_seq_len:
            raise ValueError(f"cache is full: max_seq_len={cache.max_seq_len}")
        if self.mode == "cudagraph":
            if self.graph is None:
                raise RuntimeError("call prepare() before running a cudagraph decoder")
            self.graph.replay()
            target, accepted = self.static_out
        elif self.mode.startswith("compile"):
            from .inductor_choices import wide_persistent_reductions

            with wide_persistent_reductions():  # compiles on the first call (and on any recompile)
                target, accepted = self.fn(self.model, self.tokens, cache, self.mtp_cache)
        else:
            target, accepted = self.fn(self.model, self.tokens, cache, self.mtp_cache)
        values = torch.cat([accepted, target[0]]).tolist()  # the one sync of the round
        n = values[0]
        cache.seen_tokens += n
        self.mtp_cache.seen_tokens += n
        self.stats.record(n)
        return values[1 : 1 + n]

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int) -> torch.LongTensor:
        """Greedy continuation of ``input_ids`` (``[1, P]``) by ``max_new_tokens`` tokens; ``self.stats`` holds the
        acceptance counts afterwards."""
        if input_ids.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError(f"{input_ids.shape[1]} + {max_new_tokens} tokens > max_seq_len={self.max_seq_len}")
        out = self.prefill(input_ids)[0].tolist()
        if self.graph is None:
            self.prepare()
        while len(out) < max_new_tokens:
            out += self.round()
        generated = torch.tensor(out[:max_new_tokens], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, generated.unsqueeze(0)], dim=1)
