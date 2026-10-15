# A100 Qwen3 compile 수치 오차 분리 실험

상태: **failed**.

- 실제 GPU: NVIDIA A100-SXM4-40GB, 39.4935302734375 GiB.
- PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 4.57.3.
- Qwen/Qwen3-14B, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, B=1.
- 저장된 동일 입력 128토큰, prefill 1회 + decode 127회 fixed-token replay.
- 모델은 Transformers의 공식 Qwen3 구현이다. 모델 구조나 가중치는 교체하지 않았다.
- 성능 비교가 아닌 정확도 원인 분리 실험이며, speculative decoding은 사용하지 않았다.
- 모든 비교 기준: atol=0.02, rtol=0.02, NaN/Inf 실패. 기준을 완화하지 않았다.

## 전체 모델 비교

각 행은 왼쪽 실행을 오른쪽 reference와 비교한다. logits는 전체 vocabulary, K/V는 prefill과 decode 체크포인트의 모든 레이어·전체 유효 길이다.

| 비교 | logits 실패/검사 | logits 최대 절대오차 | logits 최대 relative L2 | KV 실패/검사 | KV 최대 절대오차 | 단계별 argmax 일치 |
|---|---:|---:|---:|---:|---:|---|
| eager_static_auto / eager_dynamic_auto | 128/128 | 11.0938 | 0.458601 | 396/480 | 13.375 | True |
| compiled_static_default / eager_static_auto | 128/128 | 8.84375 | 0.313565 | 416/480 | 21.375 | True |
| compiled_static_default / eager_dynamic_auto | 128/128 | 10.7812 | 0.283599 | 418/480 | 21.5625 | True |
| compiled_static_preserve_casts / eager_static_auto | 128/128 | 15.5 | 0.402182 | 384/480 | 37.375 | True |
| compiled_static_preserve_casts / eager_dynamic_auto | 128/128 | 15.25 | 0.4568 | 396/480 | 39 | True |

`default`는 기본 Inductor 수치 경로이고, `preserve_casts`는 `emulate_precision_casts=True`다. `auto`는 SDPA 자동 선택이며 `efficient`는 memory-efficient SDPA만 허용한 제어군이다. CUDA Graph 재생은 비활성화했다.

## 실제 선택된 attention

### eager_dynamic_auto

- 실행: passed_execution.
- Prefill events: `['aten::_flash_attention_forward', 'aten::_scaled_dot_product_flash_attention', 'aten::scaled_dot_product_attention', 'void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, true, false, false, true, true, false, false>(pytorch_flash::Flash_fwd_params)']`.
- Decode events: `['aten::_flash_attention_forward', 'aten::_scaled_dot_product_flash_attention', 'aten::scaled_dot_product_attention', 'void pytorch_flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::bfloat16_t> >, 4, 1, true>(pytorch_flash::Flash_fwd_params)', 'void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::bfloat16_t> >, false, false, false, false, true, false, true, false>(pytorch_flash::Flash_fwd_params)']`.

### eager_static_auto

- 실행: passed_execution.
- Prefill events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'aten::scaled_dot_product_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)']`.
- Decode events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'aten::scaled_dot_product_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)']`.

### compiled_static_default

- 실행: passed_execution.
- Prefill events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_neg_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_5', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_6', 'triton_red_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_neg_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_10']`.
- Decode events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)', 'triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_neg_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_5', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_pow_rsqrt_scalar_tensor_transpose_unsqueeze_view_where_6']`.

### compiled_static_preserve_casts

- 실행: passed_execution.
- Prefill events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)', 'triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_neg_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_sin_transpose_unsqueeze_view_where_14', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_sin_transpose_unsqueeze_view_where_4', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_sin_transpose_unsqueeze_view_where_5']`.
- Decode events: `['aten::_efficient_attention_forward', 'aten::_scaled_dot_product_efficient_attention', 'fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)', 'triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_le_mean_mul_neg_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_sin_transpose_unsqueeze_view_where_5', 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_expand_le_mul_scalar_tensor_sin_transpose_unsqueeze_view_where_6']`.

### eager_dynamic_efficient

- 실행: running.
- Prefill events: `미수집`.
- Decode events: `미수집`.

## 첫 레이어 연산 분리

각 연산은 동일 입력·가중치를 사용해 eager와 비교한다. 생성한 GPU 코드는 `micro-*.txt`에 보존했다.

| 연산 | 기본 compile 최대 절대오차 | 기본 초과 원소 | cast 보존 최대 절대오차 | cast 보존 초과 원소 |
|---|---:|---:|---:|---:|

분리한 eager K 경로와 native 첫 레이어 K의 비교: `미완료`.

cast 보존 설정의 원래 전체 logits/KV gate 통과: **False**.

연산 분리 microtest 단계에 도달하지 못했으므로 GPU micro 생성 코드는 수집하지 않았다.

## 실행 오류

```text
RuntimeError: No available kernel. Aborting execution.
```

## 실행과 정리

- 세션: `qwen3-flash-1015-141756-5f33`.
- 실행기 경과 시간: 487.48034718300005초.
- 종료 exit code: 0.
- 서버 활성 세션 없음: True.

원자료: [diagnosis.json](diagnosis.json), [hardware.json](hardware.json), [execution.json](execution.json), [benchmark.log](benchmark.log), [lifecycle.json](lifecycle.json).
