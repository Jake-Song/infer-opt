# Qwen3.5-35B-A3B Q8_0 llama.cpp — Colab G4, MTP off

측정일: 2026-11-05 KST (원시 로그 UTC: 2026-11-04). 실제 GPU에서 실행 완료.

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.9708 GiB. CUDA 13.0.88, SM120a Release 빌드, CUDA Graph 빌드 옵션 ON.
단일 시퀀스, GPU 오프로딩 41/41 레이어, expert CPU 오프로딩 없음, Flash Attention ON, attention K/V F16, CPU threads 4, batch 2048, ubatch 512. 기본 GPU 클럭을 사용했으며 클럭을 고정하지 않았다.
MTP 및 기타 speculative decoding을 사용하지 않았다. `llama-bench`는 일반 `llama_decode`만 호출한다. 문장 생성 확인은 `--spec-type none`을 명시했다.

## 측정 결과

각 조건 워밍업 후 7회 측정한 중앙값. 디코드 출력 길이는 256토큰이다.

| 입력/기존 문맥 토큰 | 프리필 ms | 프리필 token/s | 디코드 token/s | 디코드 ms/token |
| ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | 232.52 | 4.301 |
| 512 | 48.30 | 10600.8 | 233.93 | 4.275 |
| 2,048 | 198.62 | 10310.9 | 233.90 | 4.275 |
| 8,192 | 806.12 | 10162.2 | 225.89 | 4.427 |

프리필 열은 빈 캐시에 해당 길이의 입력을 처리하는 시간이다. 디코드 열은 해당 길이로 먼저 캐시를 채운 뒤 추가 256토큰을 처리하는 시간이며, 사전 캐시 채우기와 상태 복원 시간은 제외한다.

임의 토큰을 사용하는 엔진 벤치마크다. 모델 로딩, 토큰화, 샘플링, HTTP, 워밍업을 제외한다. TTFT 또는 서비스 종단 간 지연 측정이 아니다. MoE의 expert 선택은 입력에 따라 달라지므로 실제 문장 생성 성능과 구분해야 한다. 정확도 평가, BF16 대비 수치 검증, infer-opt와의 동일 입력 비교는 수행하지 않았다.

별도 생성 확인: `Explain briefly why the sky is blue.`에 대해 Rayleigh scattering을 설명하는 정상 영어 출력을 확인했다. 48토큰 한 번의 짧은 실행에서 CLI 표시 생성 속도는 217.4 token/s였다. 반복 측정 결과와 별도로 보존하며 품질 평가나 대표 성능으로 취급하지 않는다.

## 모델 및 소스

- 배포본: https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF
- 모델 revision: `3127ef0b7fd4f12626506419ceefde54fa8db53c`
- 파일: `Qwen3.5-35B-A3B-Q8_0.gguf`, 36,903,139,584 bytes (34.3687 GiB)
- SHA-256: `8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67` (Hub LFS 해시와 일치)
- llama.cpp commit: `c6a0ff11a0fe16ac1a2eb569190934593a2b3670`
- Qwen 원본을 ggml-org가 GGUF로 배포한 Q8_0이며, infer-opt INT8 group-128 체크포인트와는 다르다.

## 재현 명령

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout c6a0ff11a0fe16ac1a2eb569190934593a2b3670
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build -j 12 --target llama-bench llama-cli
hf download ggml-org/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q8_0.gguf --revision 3127ef0b7fd4f12626506419ceefde54fa8db53c --local-dir models
```

```bash
./build/bin/llama-bench -m models/Qwen3.5-35B-A3B-Q8_0.gguf \
  -ngl 99 -fa on -t 4 -b 2048 -ub 512 -ctk f16 -ctv f16 \
  -r 7 -o jsonl -oe md -p 512,2048,8192 -n 0
./build/bin/llama-bench -m models/Qwen3.5-35B-A3B-Q8_0.gguf \
  -ngl 99 -fa on -t 4 -b 2048 -ub 512 -ctk f16 -ctv f16 \
  -r 7 -o jsonl -oe md -p 0 -n 256 -d 0,512,2048,8192
```

실제 실행은 디코드 깊이별로 별도 프로세스를 사용했다. 정확한 argv는 `results/commands.json`을 참조한다.

## 보존 파일과 정리

- `worker.py`: 원격 작업 소스; `launch.py`: Colab 실행용 소스 포함 런처.
- `results/summary.json`: 요약 및 반복별 ms; `prefill.stdout`, `decode-d*.stdout`: 원시 JSONL.
- `results/manifest.json`: 모델·소스 고정 revision 및 SHA-256.
- `results/environment.json`, `CMakeCache.txt`, `nvcc.stdout`, `gpu-before.stdout`, `gpu-after.stdout`: 환경 및 빌드.
- `results/offload-audit.stderr`: 동일 설정의 짧은 진단 실행. GPU 오프로딩과 캐시를 확인하며 성능 표에는 포함하지 않았다.
- `results/generation-smoke.stdout`: 실제 생성 결과.
- `results/`: 다운로드한 원시 결과. 모델 가중치나 clone한 저장소는 포함하지 않음.
- 이번 G4 세션 `q35moe-llama-q8` 종료 완료. 종료 뒤 `colab sessions`에서 활성 세션 없음 확인.
- 공개 llama.cpp·공개 모델·이번 벤치마크 스크립트만 사용했다. Drive 마운트, 모델 업로드, 저장소 소스 전송, commit/push는 하지 않았다.

이후 재실행용 worker/launcher는 사용자 요청에 따라 모델 다운로드를 `/tmp/infer-opt-models/huggingface/hub`로 변경했다. `results/worker.py`와 원시 manifest는 실제 측정 당시 소스/경로를 그대로 보존한다. 이번 측정 모델은 종료된 Colab VM에만 있었으며 로컬에 GGUF를 저장하지 않았다.

Git에 보존한 CMake·GPU 진단 로그는 줄 끝 공백과 파일 끝 빈 줄만 정리했다. 측정값과 내용은 변경하지 않았다.
