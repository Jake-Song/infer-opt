# Qwen3 단독 측정 기록

Baseline 성능 비교나 speculative decoding 없이 Qwen3-14B의 prefill, decode,
엔진 TTFT, CPU 수신 기준 ITL을 측정하는 작업이다. 원본 모델은 revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`로 고정한다.

- [강제 Flash SDPA + StaticCache 검사](a100-20261015-static-flash-compile/report.md):
  실제 A100 80GB에서 attention mask 비호환으로 실행 불가.
- [SDPA 자동 선택 + StaticCache + compile 검사](a100-20261015-static-compile-auto/report.md):
  모델은 실행됐으나 128-token 입력의 수치 정확도 검증 실패로 지연 측정 전 중단.
- [총 GPU 할당 감사](allocation_audit.json): 한 번 할당해 두 실행이 공유했고,
  할당 요청부터 정리 기록까지 보수적으로 약 10분 57초. 활성 세션 없음 확인.
- [수치 오차 로컬 원인 분석](local-precision-diagnosis/report.md): 실제 Qwen3 가중치로
  BF16 중간 cast 생략의 영향을 CPU에서 재현했다. 첫 레이어 K 생성 경로는 기본
  compile에서 기준 초과 6개, cast 보존 옵션에서는 eager 대비 오차 0이었다.
  A100 재할당은 서버 오류로 실패했으며, 사용자 선택에 따라 추가 GPU는 사용하지 않았다.
- [후속 A100 40GB 실험](a100-40gb-20261015-compile-diagnosis/findings.md): 이후 사용자 요청으로
  40GB에 연결했다. cast 보존 옵션이 첫 레이어 K/V의 오차를 없애는 것을 확인했지만,
  전체 logits/KV 검증은 여전히 실패했다. compile을 끈 cache/backend 변경도 차이를 보였다.
- [Transformers 4.57.3 공식 방식과 테스트 허용 오차](transformers-4.57.3-review.md):
  prefill eager + decode 자동 compile, token 중심의 생성 테스트와 별도의 BF16 attention 오차 기준을 확인했다.

## 공식 generate 기본 옵션 재검증

`--official-defaults --attention auto`는 수동 `torch.compile(model.forward)` 대신
Transformers 4.57.3의 `generate(cache_implementation="static")`와 기본 `CompileConfig()`를 사용한다.
BF16, SDPA 자동 선택을 유지하고 prefill/decode 분기는 generate에 맡긴다.
비교 재현성을 위해 greedy는 generate 키워드 인자로 명시하고, 최종 적용 설정을 저장한다.

- [수정한 greedy 실행 결과](a100-20261015-generate-defaults-greedy/report.md)
- [실행 코드](../../scripts/measure_qwen3_generate_defaults.py)
- 최초 `a100-20261015-generate-defaults` 실행은 모델 기본 sampling 설정의 덮어쓰기를 발견해
  greedy 검증 결과에서 제외했다.
- 전체 128단계 logits 및 최종 모든 레이어의 유효 K/V를 비교한다. 중간 KV checkpoint를
  비교한 이전 fixed-token replay 실험과 검사 범위가 다르다. 성능은 측정하지 않는다.
- 현재 통과 기준은 동일 입력·greedy 조건의 생성 token ID 전체 일치다. logits/KV 오차는
  진단 정보로 유지한다. 저장된 128-token 생성은 세 경로 모두 일치하여 이 기준으로 통과했다.
  원래 실행 판정은 보존하고 `generation-evaluation.json`에 재판정을 기록했다.

## 확인된 수치와 한계

128단계의 teacher-forced 전체 vocabulary logits 비교는 128/128건 실패했다.
전체 최대 절대오차는 10.53125, relative L2 최댓값은 약 0.25309다.
K/V는 480건 중 426건 실패했고, cache 길이는 255로 일치했다.
단계별 argmax는 같지만 free-generation 검증 단계까지 도달하지 않았다.

이 실행의 reference는 eager DynamicCache이며 candidate는 compiled StaticCache다.
따라서 cache·attention backend 변경과 compile의 영향을 아직 분리하지 못했다.
첫 레이어 K에도 차이가 있어 컴파일의 BF16 연산 반올림 변화가 원인 후보지만
확정된 전체 GPU 원인은 아니다. 이후 로컬 연산 분리에서 BF16 cast 생략의 영향을
재현했으며 상세 근거와 제한은 위 로컬 분석 보고서에 있다. 후속 코드에는 실패 시
eager StaticCache 대조 검사와 `--emulate-precision-casts` 옵션을 추가했다.
**수정된 경로는 아직 GPU에서 실행하지 않았다.**
PyTorch 2.11의 해당 옵션은 fused BF16 연산에서 eager의 중간 cast 경계를
보존하기 위한 기능이다. [PyTorch 소스](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/config.py)

## 후속 재현 명령

아래 명령은 **새 GPU를 할당**하며 이 기록을 작성할 때는 실행하지 않았다.
이전 약 11분과 합쳐 2시간을 넘지 않도록 새 실행 상한을 100분으로 둔다.
Transformers 4.57.3을 설치하되 Colab의 CUDA PyTorch는 교체하지 않고 Drive도 사용하지 않는다.

```bash
.venv/bin/python scripts/run_qwen3_flash_compile_colab.py \
  --attention auto --emulate-precision-casts --budget-minutes 100 \
  --output experiments/qwen3_flash_compile/a100-static-compile-preserve-casts
```

정확도 기준을 완화하지 않는다. 검증을 통과해야 길이별 warmup 2회와
측정 3회를 수행하고, profiler는 별도 실행한다. 새 실행은 성공·실패 모두
결과 회수 후 자신이 사용한 세션을 종료한다. 연결 오류에 대비해 원격 실행은
노트북 WebSocket과 분리한 supervisor가 관리한다.

저장된 결과의 보고서는 GPU 연결 없이 다시 만들 수 있다.

```bash
.venv/bin/python scripts/summarize_qwen3_measurement.py \
  experiments/qwen3_flash_compile/a100-20261015-static-compile-auto
```

로컬 확인: 세 실행 스크립트의 Python 구문 검사, CLI help, 보고서 생성,
`git diff --check` 통과. 실제 GPU 검증은 위 두 기록의 범위에 한한다.
