import dataclasses
import json

import pytest
import torch

from infer_opt.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from infer_opt.speculative import MTPSpeculativeDecoder, spec_round


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=[1, 1, 0], mtp_num_hidden_layers=1,
    )


def make_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config(), mtp=True).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "norm" in name or name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * 0.2)
    return model


def linear_states(cache):
    return [t for t in cache.conv_states + cache.recurrent_states if t is not None]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_speculative_greedy_matches_greedy(k):
    model = make_model()
    ids = torch.randint(0, 97, (1, 11))
    want = model.generate_greedy(ids, 30)
    decoder = MTPSpeculativeDecoder(model, max_seq_len=41, k=k)
    assert torch.equal(decoder.generate(ids, 30), want)
    assert decoder.stats.rounds >= 29 / (k + 1)
    assert torch.equal(decoder.generate(ids, 30), want)  # a second sequence on the same decoder


@pytest.mark.parametrize("accepted", [1, 2, 4])
def test_verify_then_commit_matches_sequential_decode(accepted):
    model = make_model()
    ids = torch.randint(0, 97, (1, 30))
    tokens = 4
    with torch.no_grad():
        seq, spec = model.new_cache(1, 40), model.new_cache(1, 40)
        spec.allocate_spec_buffers(tokens)
        model(ids[:, :20], seq)
        model(ids[:, :20], spec)
        want = torch.cat([model.decode_step(ids[:, i : i + 1], seq) for i in range(20, 20 + accepted)], dim=1)
        got = model.lm_head(model.model.language_model.forward_verify(ids[:, 20 : 20 + tokens], spec))
        assert torch.allclose(got[:, :accepted], want, atol=1e-10)
        assert spec.position.item() == 20  # nothing committed yet
        spec.commit(torch.tensor([accepted]))
        spec.seen_tokens += accepted
        assert spec.position.item() == seq.position.item() == 20 + accepted
        for a, b in zip(linear_states(spec), linear_states(seq)):
            assert torch.allclose(a, b, atol=1e-12)
        # the rejected tokens' K/V past the position must not leak into what follows
        nxt = ids[:, 25:26]
        assert torch.allclose(model.decode_step(nxt, spec), model.decode_step(nxt, seq), atol=1e-10)


def test_round_accepts_true_drafts_and_rejects_wrong_ones():
    model = make_model()
    ids = torch.randint(0, 97, (1, 15))
    greedy = model.generate_greedy(ids, 8)[0, 15:]
    for drafts, expect in ((greedy[1:4], 4), ((greedy[1:4] + 1) % 97, 1), (torch.stack([greedy[1], (greedy[2] + 1) % 97, greedy[3]]), 2)):
        decoder = MTPSpeculativeDecoder(model, max_seq_len=30, k=3)
        assert decoder.prefill(ids).item() == greedy[0]
        decoder.tokens[0, 1:] = drafts
        with torch.no_grad():
            target, accepted = spec_round(model, decoder.tokens, decoder.cache, decoder.mtp_cache)
        assert accepted.item() == expect
        assert torch.equal(target[0, :expect], greedy[1 : 1 + expect])
        assert decoder.tokens[0, 0] == greedy[expect]  # the next round starts from the last accepted token
        assert decoder.cache.position.item() == decoder.mtp_cache.position.item() == 15 + expect


def test_mtp_prefill_cache_matches_fixed_shape_steps():
    # the MTP KV a round writes for the accepted positions equals the one a prefill over the same pairs writes
    model = make_model()
    ids = torch.randint(0, 97, (1, 12))
    hidden = torch.randn(1, 12, 64, dtype=torch.float64)
    with torch.no_grad():
        full = model.new_mtp_cache(16)
        want = model.mtp_hidden(ids, hidden, full)
        part = model.new_mtp_cache(16)
        model.mtp_hidden(ids[:, :8], hidden[:, :8], part)
        got = model.mtp_hidden(ids[:, 8:], hidden[:, 8:], part, decode=True)
    assert torch.allclose(got, want[:, 8:], atol=1e-10)
    assert torch.allclose(part.keys[0][:, :, :12], full.keys[0][:, :, :12], atol=1e-12)


def test_decoder_needs_mtp_and_checks_lengths():
    model = make_model()
    with pytest.raises(ValueError, match="MTP"):
        MTPSpeculativeDecoder(Qwen3_5ForCausalLM(tiny_config()), max_seq_len=10)
    decoder = MTPSpeculativeDecoder(model, max_seq_len=10, k=2)
    with pytest.raises(ValueError, match="max_seq_len"):
        decoder.generate(torch.randint(0, 97, (1, 6)), 5)
    with pytest.raises(ValueError, match="one sequence"):
        decoder.prefill(torch.randint(0, 97, (2, 6)))


def test_load_mtp_head_from_checkpoint_dir(tmp_path):
    from safetensors.torch import save_file

    model = make_model().float()
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}  # tied
    assert {"mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight", "mtp.pre_fc_norm_hidden.weight"} <= set(state)
    save_file(state, str(tmp_path / "model.safetensors"))
    c = model.config
    text = {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
    text["rope_parameters"] = {"rope_theta": text.pop("rope_theta"), "mrope_section": text.pop("mrope_section"),
                               "partial_rotary_factor": text.pop("partial_rotary_factor")}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))

    plain = Qwen3_5ForCausalLM.from_pretrained_dir(tmp_path, dtype=torch.float32)
    assert plain.mtp is None and not any(k.startswith("mtp.") for k in plain.state_dict())
    loaded = Qwen3_5ForCausalLM.from_pretrained_dir(tmp_path, dtype=torch.float32, mtp=True)
    assert loaded.config.mtp_num_hidden_layers == 1
    ids = torch.randint(0, 97, (1, 9))
    with torch.no_grad():
        assert torch.equal(loaded(ids), model(ids))
        h = torch.randn(1, 9, 64)
        assert torch.equal(loaded.mtp_hidden(ids, h, loaded.new_mtp_cache(9)), model.mtp_hidden(ids, h, model.new_mtp_cache(9)))


def test_fused_projections_cover_the_mtp_layer():
    model = make_model()
    ids = torch.randint(0, 97, (1, 9))
    want = model.generate_greedy(ids, 12)
    assert model.fuse_projections() == 10  # the target's 8 groups and the MTP layer's attention and MLP
    assert torch.equal(MTPSpeculativeDecoder(model, max_seq_len=21, k=2).generate(ids, 12), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton decode attention needs CUDA")
@pytest.mark.parametrize("position", [0, 61, 86])  # empty cache, tokens straddling a 64-key slice, last slots
def test_cuda_multi_token_decode_attention_matches_fp32_reference(position):
    from infer_opt.attention_kernels import decode_attention

    torch.manual_seed(0)
    tokens = 4
    q = torch.randn(2, 4, 6, tokens, 256, device="cuda", dtype=torch.float16)  # 27B layout: 6 query heads per KV head
    k, v = (torch.randn(2, 4, 90, 256, device="cuda", dtype=torch.float16) for _ in range(2))
    pos = torch.tensor([position], device="cuda")
    scores = (q.float().unsqueeze(-2) * k.float()[:, :, None, None]).sum(-1) / 16  # [B, KV, G, T, L]
    last = pos + torch.arange(tokens, device="cuda")
    scores = scores.masked_fill(torch.arange(90, device="cuda") > last[:, None], float("-inf"))
    expected = (scores.softmax(-1).unsqueeze(-1) * v.float()[:, :, None, None]).sum(-2)
    query = q.reshape(2, 4, 6 * tokens, 256)
    for fn in (decode_attention, torch.compile(decode_attention, fullgraph=True)):
        got = fn(query, k, v, pos, 1 / 16, tokens).view(2, 4, 6, tokens, 256)
        assert got.dtype == torch.float16 and torch.allclose(got.float(), expected, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton GEMV needs CUDA")
@pytest.mark.parametrize("form", [None, ("fma", 4, 256, 4), ("fma", 1, 128, 2), ("dot", 32, 4, 2), ("dot", 16, 2, 2, 4)])
@pytest.mark.parametrize("rows", [2, 3, 8])
@pytest.mark.parametrize("out_f,in_f,group_size", [(300, 384, 128), (640, 5120, 128), (200, 384, 48), (96, 256, 64)])
def test_cuda_group_gemv_rows_match_dequantized_matmul_and_single_rows(monkeypatch, form, rows, out_f, in_f,
                                                                         group_size):
    from infer_opt import quant_kernels
    from infer_opt.quant import Int8Linear

    if form is not None and form[0] == "fma" and in_f % form[2]:
        pytest.skip("BLOCK_K must divide K")
    if form is not None and len(form) > 4 and (in_f // group_size) % form[4]:
        pytest.skip("the split must divide the number of groups")
    if form is not None:  # None: the default rule for this GPU
        monkeypatch.setitem(quant_kernels.ROWS_LAUNCH_CONFIGS, (rows, out_f, in_f, group_size), form)
    torch.manual_seed(0)
    linear = torch.nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
    q = Int8Linear.from_linear(linear, group_size=group_size)  # group 48: not a power of two, dequantized matmul
    x = torch.randn(1, rows, in_f, device="cuda", dtype=torch.float16)
    got = q(x)
    expected = torch.nn.functional.linear(x.float(), q.dequantized_weight().float())
    assert got.shape == (1, rows, out_f) and torch.allclose(got.float(), expected, atol=2e-2, rtol=2e-3)
    single = torch.cat([quant_kernels.w8a16_group_gemv(x[:, i : i + 1], q.weight, q.scale) for i in range(rows)], 1)
    assert torch.allclose(got.float(), single.float(), atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("mode", ["cudagraph", "compile"])
def test_cuda_speculative_modes_match_eager(mode):
    model = make_model().float().cuda()
    ids = torch.randint(0, 97, (1, 20), device="cuda")
    eager = MTPSpeculativeDecoder(model, max_seq_len=60, k=2)
    want = eager.generate(ids, 40)
    torch._dynamo.reset()
    fast = MTPSpeculativeDecoder(model, max_seq_len=60, k=2, mode=mode)
    assert torch.equal(fast.generate(ids, 40), want)
    assert fast.stats.as_dict() == eager.stats.as_dict()
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel")
@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_gated_delta_verify_matches_sequential_steps(tokens, dtype):
    from infer_opt.gated_delta_kernels import gated_delta_verify
    from infer_opt.qwen3_5 import torch_recurrent_gated_delta_rule

    torch.manual_seed(0)
    batch, heads, value_heads, dk, dv = 2, 4, 12, 64, 32  # 3 value heads per query/key head, like the 27B
    q, k = (torch.randn(batch, tokens, heads, dk, device="cuda", dtype=dtype) for _ in range(2))
    v = torch.randn(batch, tokens, value_heads, dv, device="cuda", dtype=dtype)
    g = -torch.rand(batch, tokens, value_heads, device="cuda")  # log decay, fp32 as in the model
    beta = torch.rand(batch, tokens, value_heads, device="cuda", dtype=dtype)
    h0 = torch.randn(batch, value_heads, dk, dv, device="cuda")
    states = torch.empty(tokens, batch, value_heads, dk, dv, device="cuda")
    out = gated_delta_verify(q, k, v, g, beta, h0, states)
    state = h0
    rep = value_heads // heads
    for t in range(tokens):
        step = slice(t, t + 1)
        want, state = torch_recurrent_gated_delta_rule(
            q[:, step].repeat_interleave(rep, 2).float(), k[:, step].repeat_interleave(rep, 2).float(),
            v[:, step].float(), g=g[:, step], beta=beta[:, step].float(), initial_state=state,
            use_qk_l2norm_in_kernel=True)
        assert torch.allclose(out[:, step].float(), want, atol=2e-3 if dtype == torch.float16 else 1e-5), t
        assert torch.allclose(states[t], state, atol=1e-5), t
    assert out.dtype == dtype and out.shape == v.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused verify kernel")
@pytest.mark.parametrize("accepted", [1, 3])
def test_cuda_verify_then_commit_matches_sequential_decode(accepted):
    model = make_model().float().cuda()
    ids = torch.randint(0, 97, (1, 30), device="cuda")
    with torch.no_grad():
        seq, spec = model.new_cache(1, 40), model.new_cache(1, 40)
        spec.allocate_spec_buffers(4)
        model(ids[:, :20], seq)
        model(ids[:, :20], spec)
        want = torch.cat([model.decode_step(ids[:, i : i + 1], seq) for i in range(20, 20 + accepted)], dim=1)
        got = model.lm_head(model.model.language_model.forward_verify(ids[:, 20:24], spec))
        assert torch.allclose(got[:, :accepted], want, atol=1e-4)
        spec.commit(torch.tensor([accepted], device="cuda"))
        spec.seen_tokens += accepted
        for a, b in zip(linear_states(spec), linear_states(seq)):
            assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel")
def test_cuda_gated_delta_verify_matches_fla_steps():
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    from infer_opt.gated_delta_kernels import gated_delta_verify

    torch.manual_seed(0)
    batch, tokens, heads, value_heads, dk, dv = 1, 4, 16, 48, 128, 128  # the 27B layer
    q, k = (torch.randn(batch, tokens, heads, dk, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(batch, tokens, value_heads, dv, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, tokens, value_heads, device="cuda")
    beta = torch.rand(batch, tokens, value_heads, device="cuda", dtype=torch.bfloat16)
    h0 = torch.randn(batch, value_heads, dk, dv, device="cuda")
    states = torch.empty(tokens, batch, value_heads, dk, dv, device="cuda")
    out = gated_delta_verify(q, k, v, g, beta, h0, states)
    state = h0
    for t in range(tokens):  # what verifying did before: one FLA decode step per token
        step = slice(t, t + 1)
        want, state = fla.fused_recurrent_gated_delta_rule(
            q[:, step].contiguous(), k[:, step].contiguous(), v[:, step].contiguous(), g=g[:, step],
            beta=beta[:, step], initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        assert torch.allclose(out[:, step].float(), want.float(), atol=1e-2, rtol=1e-2), t
        assert torch.allclose(states[t], state, atol=1e-5, rtol=1e-5), t


def test_read_safetensors_reads_only_the_prefix(tmp_path):
    from safetensors.torch import save_file

    from infer_opt.hub import read_safetensors

    tensors = {"mtp.fc.weight": torch.randn(3, 4).bfloat16(), "mtp.norm.weight": torch.arange(5, dtype=torch.float32),
               "model.x": torch.ones(2, dtype=torch.int8), "mtp.empty": torch.empty(0, 3)}
    save_file(tensors, str(tmp_path / "s.safetensors"), metadata={"format": "pt"})
    with open(tmp_path / "s.safetensors", "rb") as f:
        got = read_safetensors(f, "mtp.")
    assert set(got) == {"mtp.fc.weight", "mtp.norm.weight", "mtp.empty"}
    for name, t in got.items():
        assert t.dtype == tensors[name].dtype and torch.equal(t, tensors[name])


def test_attach_mtp_to_a_quantized_checkpoint(tmp_path):
    from infer_opt.quant import quantize_linears, save_quantized

    source = make_model().float()
    mtp_state = {f"mtp.{k}": v for k, v in source.mtp.state_dict().items()}
    target = Qwen3_5ForCausalLM(tiny_config()).float().eval()  # the same weights, without the MTP head
    target.load_state_dict({k: v for k, v in source.state_dict().items() if not k.startswith("mtp.")})
    quantize_linears(target, group_size=32, min_out_features=16)
    save_quantized(target, tmp_path)
    c = target.config
    text = {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
    text["rope_parameters"] = {"rope_theta": text.pop("rope_theta"), "mrope_section": text.pop("mrope_section"),
                               "partial_rotary_factor": text.pop("partial_rotary_factor")}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))

    loaded = Qwen3_5ForCausalLM.from_quantized_dir(tmp_path, dtype=torch.float32)
    assert loaded.mtp is None
    loaded.attach_mtp(mtp_state)
    ids = torch.randint(0, 97, (1, 10))
    with torch.no_grad():
        h = torch.randn(1, 10, 64)
        assert torch.equal(loaded.mtp_hidden(ids, h, loaded.new_mtp_cache(10)),
                           source.mtp_hidden(ids, h, source.new_mtp_cache(10)))
    assert torch.equal(MTPSpeculativeDecoder(loaded, max_seq_len=30, k=2).generate(ids, 20),
                       loaded.generate_greedy(ids, 20))
