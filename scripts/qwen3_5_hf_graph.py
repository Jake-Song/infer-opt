"""HF-only batch-1 decode helpers, embedded in the standalone 27B Colab notebook.

Targets transformers==5.20.2. The model and cache implementations are unmodified
Hugging Face code. Only the decode scheduling and greedy token selection live here.
"""

from __future__ import annotations

import time

import torch
from transformers import DynamicCache, StaticCache


def cache_tensors(cache):
    """Include both attention KV and GatedDeltaNet state/counters."""
    result = {}
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values", "cumulative_length", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            values = value.items() if isinstance(value, dict) else [(None, value)]
            for key, tensor in values:
                if isinstance(tensor, torch.Tensor):
                    result[f"{i}.{name}.{key}"] = tensor
    return result


class HFGraphDecoder:
    def __init__(self, model, max_cache_len):
        self.model = model
        self.device = next(model.parameters()).device
        self.max_cache_len = max_cache_len
        self.cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        self.token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.position = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.graph = None
        self.compiled = None
        self.addresses = None
        self.used = 0

    def check_addresses(self):
        current = {k: t.data_ptr() for k, t in cache_tensors(self.cache).items()}
        if self.addresses is not None and current != self.addresses:
            raise RuntimeError("Cache storage changed after capture; rebuild the decoder.")
        return current

    @torch.inference_mode()
    def prefill(self, ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] < self.max_cache_len:
            raise ValueError("Expected one nonempty, unpadded prompt shorter than max_cache_len.")
        self.cache.reset()
        output = self.model(input_ids=ids, past_key_values=self.cache,
                            use_cache=True, logits_to_keep=1)
        self.token.copy_(output.logits[:, -1].argmax(-1, keepdim=True))
        self.position.fill_(ids.shape[1])
        self.used = ids.shape[1]
        self.check_addresses()
        return output.logits

    def _decode(self):
        logits = self.model(input_ids=self.token, position_ids=self.position,
                            past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1).logits
        self.token.copy_(logits[:, -1].argmax(-1, keepdim=True))
        self.position.add_(1)
        return logits

    @torch.inference_mode()
    def prepare(self, ids):
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph preparation requires a CUDA GPU.")
        if ids.shape[1] + 4 > self.max_cache_len:
            raise ValueError("Leave at least four cache positions for compile/capture warmup.")
        if self.graph is not None:
            raise RuntimeError("Already captured. Reuse prefill(), or create a new decoder.")
        self.prefill(ids)  # initialize every KV/conv/recurrent buffer before compiling
        self.addresses = self.check_addresses()
        # Disable automatic graphs: explicit capture below guarantees graph replay,
        # including all cache mutations, instead of silently skipping mutated inputs.
        self.compiled = torch.compile(self._decode, fullgraph=True, dynamic=False,
                                      mode="max-autotune-no-cudagraphs")
        torch.cuda.synchronize()
        start = time.perf_counter()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.compiled()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start
        self.check_addresses()

        self.prefill(ids)  # warmup consumed tokens and changed recurrent state
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_logits = self.compiled()
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - start
        self.check_addresses()
        self.prefill(ids)  # capture must not become part of the real sequence
        return {"compile_warmup_seconds": compile_seconds,
                "capture_seconds": capture_seconds, "explicit_cuda_graph": True}

    @torch.inference_mode()
    def step(self, forced_token=None, mode="graph"):
        if self.used >= self.max_cache_len:
            raise ValueError("Cache capacity exceeded; reset with prefill().")
        if forced_token is not None:
            self.token.copy_(forced_token)
        if mode == "graph":
            if self.graph is None:
                raise RuntimeError("Call prepare() before graph replay.")
            self.graph.replay()
            logits = self.graph_logits
        elif mode == "compiled":
            if self.compiled is None:
                raise RuntimeError("Call prepare() before compiled decode.")
            logits = self.compiled()
        elif mode == "eager":
            logits = self._decode()
        else:
            raise ValueError(f"Unknown decode mode: {mode}")
        self.used += 1
        return logits  # graph output is overwritten by the next replay


@torch.inference_mode()
def validate_decoder(decoder, ids, steps=16, modes=("eager", "compiled", "graph"),
                     max_mean_kl=0.005, margin_threshold=0.5):
    """Teacher-force the same HF dynamic-cache tokens through all decode paths."""
    if ids.shape[1] + steps > decoder.max_cache_len:
        raise ValueError("Validation sequence exceeds cache capacity.")
    model = decoder.model
    reference_cache = DynamicCache(config=model.config)
    output = model(input_ids=ids, past_key_values=reference_cache,
                   use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    fed, reference = [], []
    for _ in range(steps):
        fed.append(token.clone())
        output = model(input_ids=token, past_key_values=reference_cache,
                       use_cache=True, logits_to_keep=1)
        reference.append(output.logits[:, -1].float().cpu())
        token = output.logits[:, -1].argmax(-1, keepdim=True)
    del output, reference_cache
    reference = torch.cat(reference)
    ref_logp = reference.log_softmax(-1)
    top2 = reference.topk(2, dim=-1)
    margins = top2.values[:, 0] - top2.values[:, 1]
    rows = []
    for mode in modes:
        # Repeat graph validation from a reset cache to check replay after a new prefill.
        for repeat in range(2 if mode == "graph" else 1):
            decoder.prefill(ids)
            got = torch.cat([decoder.step(t, mode=mode)[:, -1].float().cpu() for t in fed])
            mismatch = got.argmax(-1) != reference.argmax(-1)
            kl = (ref_logp.exp() * (ref_logp - got.log_softmax(-1))).sum(-1)
            strong = int((mismatch & (margins > margin_threshold)).sum())
            finite = bool(torch.isfinite(got).all() and torch.isfinite(kl).all())
            length_ok = int(decoder.cache.get_seq_length()) == ids.shape[1] + steps
            position_ok = int(decoder.position.item()) == ids.shape[1] + steps
            row = {"mode": mode, "reset_repeat": repeat, "steps": steps,
                   "mean_kl": float(kl.mean()), "max_abs_logit_diff": float((got - reference).abs().max()),
                   "top1_agreement": float((~mismatch).float().mean()), "strong_mismatches": strong,
                   "cache_length_ok": length_ok, "position_ok": position_ok,
                   "passed": finite and length_ok and position_ok and strong == 0
                   and float(kl.mean()) <= max_mean_kl}
            rows.append(row)
            decoder.check_addresses()
    return rows


@torch.inference_mode()
def benchmark_decoder(decoder, ids, steps=128, warmup=3, repeats=5):
    """Wall time includes graph replay dispatch + argmax; excludes prefill/warmup."""
    if steps < 1 or repeats < 1 or ids.shape[1] + warmup + steps > decoder.max_cache_len:
        raise ValueError("Invalid benchmark length or cache capacity.")
    samples = []
    for _ in range(repeats):
        decoder.prefill(ids)
        for _ in range(warmup):
            decoder.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(steps):
            decoder.step()
        end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        samples.append({"wall_ms_per_token": elapsed * 1000 / steps,
                        "cuda_ms_per_token": begin.elapsed_time(end) / steps,
                        "decode_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "decode_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
    return samples


@torch.inference_mode()
def generate_text(decoder, tokenizer, prompt, max_new_tokens=160):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(decoder.device)
    if max_new_tokens < 1 or ids.shape[1] + max_new_tokens > decoder.max_cache_len:
        raise ValueError("Prompt + requested output exceeds cache capacity.")
    decoder.prefill(ids)
    eos = decoder.model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    tokens = []
    for i in range(max_new_tokens):
        if i:
            decoder.step()
        token = int(decoder.token.item())
        tokens.append(token)
        if token in stop_ids:
            break
    return {"prompt_tokens": ids.shape[1], "new_tokens": len(tokens), "token_ids": tokens,
            "text": tokenizer.decode(tokens, skip_special_tokens=True)}
