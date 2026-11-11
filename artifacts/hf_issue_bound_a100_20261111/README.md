# HF Jobs A100 counter 권한 진단 — 2026-11-11

**HF Jobs에서도 이번 A100 컨테이너는 GPU performance counter에 접근하지 못했다. Instruction-issue bound 가설은 이 실행으로 확정하지 못했다.** MoE 전체 빌드 전에 작은 CUDA 커널로 권한을 검사했으며, counter 수집 실패를 확인한 뒤 추가 GPU Job을 제출하지 않았다.

사용자 지시에 따라 배포 전에 HF CLI를 **1.16.4에서 2.4.0으로 업데이트**했다. `hf update`는 설치 방식을 인식하지 못해 실패했고, 기존 `/home/jake/.hf-cli/venv`의 `huggingface_hub`를 `uv pip install --upgrade`로 갱신했다. 로그인 유지와 새 Jobs 옵션을 확인한 후 실행했다.

## 실제 실행

- [Job](https://huggingface.co/jobs/Jakemu/6af41296ab34849891ade37a), ID `6af41296ab34849891ade37a`, namespace `Jakemu`, `a100-large`, timeout 10분, attempts 1.
- 실제 GPU **NVIDIA A100-SXM4-80GB**, driver **580.178.04**, CUDA 13.0.
- 이미지 `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`를 HF Jobs에서 실행했다. 이미지 이름이 RunPod여도 이번 실행 제공자는 HF Jobs다.
- Root (`uid=0`)로 CUDA 커널을 컴파일하고 실행했다. 모델·가중치 다운로드는 없다.
- 1M 원소에 64회 FMA를 수행하는 작은 kernel을 5회 호출하고, `ncu --set basic --launch-skip 3 --launch-count 1 --csv`로 한 번의 launch를 수집하려 했다.

## 확인한 권한과 결과

| 항목 | 실제 값 |
| --- | --- |
| 실행 UID | `0` |
| Host NVIDIA driver | `RmProfilingAdminOnly: 1` |
| Effective / permitted / bounding capabilities | 모두 `00000000a80405fb` |
| `CAP_SYS_ADMIN` | 세 capability 집합 모두 없음 |
| `CAP_PERFMON` | 세 capability 집합 모두 없음 |
| CUDA compile / device execution | 성공, 각각 status `0` |
| Nsight Compute | exit `1`, `ERR_NVGPUCTRPERM` |

[permissions.json](probe/results/permissions.json), [ncu 오류 원문](probe/results/ncu-probe.stdout), [진단 결과](probe/results/result.json)에 원본을 보존했다. 따라서 CUDA 코드 실행 실패나 compiler 설치 실패가 아니라, 실제 counter 접근 권한 부족을 확인한 실행이다.

[NVIDIA 공식 설명](https://developer.nvidia.com/ERR_NVGPUCTRPERM)에 따르면 container profiling에는 host에서의 접근 허용 또는 적절한 capability가 필요하다. 이번 컨테이너는 host가 admin-only이고 필요한 capability가 bounding set에도 없어, 단순히 `sudo`나 root 계정으로 다시 실행해서 해결할 조건이 아니다.

업데이트한 HF CLI 2.4.0의 Jobs 실행 옵션, Python `HfApi.run_job` signature 및 [공개 Jobs 설정 문서](https://huggingface.co/docs/hub/jobs-configuration)에서 container capability를 추가하는 지원 옵션은 확인하지 못했다. 모든 HF 호스트나 별도 기업 계약의 설정 가능성을 부정하는 결론은 아니다. Host 설정 변경이나 플랫폼 측 profiling 권한 제공 없이는 현재 실행 설정으로 issue·DRAM/L2·stall counter를 수집할 수 없다.

## 검산과 종료

Job은 `COMPLETED`로 종료됐다. 이는 진단 스크립트가 오류를 기록하고 결과를 회수 가능한 형태로 내보냈다는 뜻이며, counter 수집 성공을 뜻하지 않는다. `counter_collection_succeeded=false`와 profiler exit 1을 따로 저장했다.

결과 tar.gz를 Job 로그에서 회수하고 SHA-256을 검산했다. 제출한 코드·request hash·Job metadata·원시 로그·종료 상태를 `probe/`에 남겼다. 실행기와 재현용 probe는 [scripts/hf_issue_check](../../scripts/hf_issue_check/)에 있다. 자동 종료된 Job이라 계속 실행 중인 GPU 세션은 남기지 않았다.

다음 원인 분석에는 profiling 권한이 허용된 실행 환경이 필요하다. 권한 없는 counter 오류를 issue-bound의 증거로 사용하지 않는다. 이전 hot/cold down 측정은 HBM만으로 병목을 설명하기 어렵다는 정황이며, instruction issue·execution pipeline·L2 대역폭·의존성·occupancy를 구분하는 counter 결과는 아직 없다.
