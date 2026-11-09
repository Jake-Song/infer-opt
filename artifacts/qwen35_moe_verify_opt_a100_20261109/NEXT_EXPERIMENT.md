# 다음 시험: expert별 route pack + Grouped GEMM

사용자가 다음 시험으로 남기도록 요청했다. 아직 시작하지 않았다.

- 대상: Qwen3.5-35B-A3B INT8 group128, parallel target verification T=2/4, top-k=8, A100.
- GPU에서 route를 expert별로 정렬/묶고, count·offset·원래 route index를 만든다. 같은 expert의 activation을 연속 버퍼로 pack한다.
- Gate/up와 down에 variable-M grouped GEMM을 적용하고, 유효한 expert/tile만 처리하는 persistent scheduler를 검토한다. Singleton GEMV + 여러 토큰 GEMM의 혼합도 비교한다.
- 결과를 원래 token/slot 순서로 복원하고, 기존 BF16 expert 가중합 순서를 보존한다.
- 비교 시간에는 pack·offset 생성·양자화/역양자화·복원·combine 비용을 모두 포함한다. Expert 단독과 target body 전체를 분리한다.
- 실제 route와 fully-shared/disjoint 대조를 모두 사용한다. T=1..5·동적 routes·compile/CUDA Graph를 확인한다.
- Tensor Core 입력 dtype·scale 적용·reduction 순서가 바뀔 경우 hidden/logits·active state·greedy ID 및 품질을 따로 검증한다. 토큰 ID 몇 개의 일치만으로 동등성을 확정하지 않는다.
- 현재 1-warp pair-load 및 기존 4-warp GEMV를 모두 기준으로 비교한다. 이전 16행 padding/gather 초안의 실패를 Grouped GEMM 전체의 결론으로 취급하지 않는다.

소스 참고: [llama.cpp 분석](../llamacpp_qwen35_moe_mtp_source_review_20261109/README.md). A100 Q8_0 T<=8 기본 경로는 W8A8 MMVQ이며, expert별 pack/양자화/scatter와 MMQ는 별도 큰 배치 경로다. 새 실험은 시작하지 않았다.
