#!/usr/bin/env python3
"""Render saved Qwen3 measurement artifacts without connecting to a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(output: Path) -> str:
    def read(name, default):
        path = output / name
        return json.loads(path.read_text()) if path.exists() else default

    execution = read("execution.json", {})
    lifecycle = read("lifecycle.json", {})
    hardware = read("hardware.json", read("allocation_hardware.json", {}))
    rows = read("measurements.json", [])
    lines = ["# Qwen3-14B StaticCache + torch.compile 측정", "",
             f"실행 상태: **{execution.get('status', lifecycle.get('status', 'unknown'))}**.", "",
             "Baseline 성능 비교 및 speculative decoding 없이 target 모델 하나만 실행했다.", "",
             f"- GPU: {hardware.get('gpu', '미확인')}; "
             f"메모리 {hardware.get('gpu_memory_gib', hardware.get('memory_gib', '미확인'))} GiB.",
             f"- PyTorch {hardware.get('torch', '미확인')}, CUDA {hardware.get('cuda', '미확인')}, "
             f"Transformers {hardware.get('transformers', '미확인')}.",
             "- 요청 조건: Qwen/Qwen3-14B, BF16, B=1, 고정 합성 토큰 입력, greedy 출력 128토큰.",
             f"- Revision: `{execution.get('revision', '미확인')}`.",
             f"- SDPA 정책: `{execution.get('attention', '미확인')}`. "
             "자동 선택은 Flash Attention 사용을 보장하지 않는다.",
             "- 전체 forward compile: fullgraph=True, dynamic=False, triton.cudagraphs=False.",
             "- 길이별 최대 cache 용량은 입력 길이 + 128이며, 길이별 별도 컴파일·warmup을 수행한다.", ""]
    if execution.get("error"):
        lines += ["실행 오류:", "", "```text", execution["error"], "```", ""]
    lines += ["## 측정값", "",
              "정확도와 greedy 출력 검증을 통과한 길이만 성능값을 표시한다. "
              "설정은 길이별 warmup 2회, 측정 3회이며 완료한 값만 요청별 중앙값으로 표시한다.", "",
              "| 입력 | Prefill ms | TTFT ms | Decode 총 ms | Decode ms/token | token/s | ITL p50 ms | ITL p95 ms |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("status") != "passed" or "summary" not in row:
            lines.append(f"| {row['prompt_len']} | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |")
            continue
        s = row["summary"]
        lines.append(f"| {row['prompt_len']} | {s['prefill_ms']:.3f} | {s['ttft_ms']:.3f} | "
                     f"{s['decode_ms']:.3f} | {s['decode_ms_per_token']:.3f} | "
                     f"{1000 / s['decode_ms_per_token']:.2f} | {s['itl_p50_ms']:.3f} | {s['itl_p95_ms']:.3f} |")
    if not rows:
        lines += ["", "본 모델의 성능 측정값은 생성되지 않았다."]
    lines += ["", "Prefill은 KV 기록과 마지막 위치의 vocabulary logits 계산까지이며 토큰 선택은 제외한다. "
              "TTFT는 미리 할당된 cache의 reset 시작부터 첫 토큰 CPU 수신까지다. "
              "Cache 최초 할당·모델 로딩·토큰화·전송·컴파일·정확도 검증·warmup은 정상 상태 지연에서 제외한다.", "",
              "Decode는 첫 토큰 이후 127회 생성의 동기화된 wall time이다. "
              "ITL은 별도 streaming 실행에서 토큰을 하나씩 CPU로 읽은 시각의 차이며, "
              "네트워크 시간은 제외한다. ITL p50/p95는 길이별 3×127개 간격을 모아 계산한다.", "",
              "## 정확도·준비 비용·실제 attention", ""]
    for row in rows:
        validation = row.get("validation", {})
        checks = validation.get("logits", []) + validation.get("kv", [])
        lines += [f"### 입력 {row['prompt_len']}", "",
                  f"- 수치 검증: {validation.get('passed', '미완료')}; "
                  f"전체 vocabulary logits 및 모든 레이어의 유효 K/V 비교 {len(checks)}건.",
                  f"- 같은 토큰을 넣었을 때 단계별 argmax 일치: {validation.get('teacher_forced_argmax_equal', '미확인')}. "
                  "이것만으로 free-generation 일치나 logits/KV 정확도를 보장하지 않는다.",
                  f"- logits 최대 절대오차: {max((r['max_abs_error'] for r in validation.get('logits', [])), default=0):.6g}; "
                  f"relative L2 최댓값: {max((r['relative_l2'] for r in validation.get('logits', [])), default=0):.6g}.",
                  f"- compile + 정확도 검증: {row.get('compile_and_validation_seconds', '미완료')}초. "
                  "순수 compiler 시간만 분리한 수치는 아니다.",
                  f"- warmup: {row.get('warmup_and_decode_compile_seconds', '미완료')}초.",
                  f"- 측정 중 compile counter 변화: {row.get('recompiled_during_measurement', '미확인')}.",
                  f"- Prefill attention profiler events: `{row.get('prefill_attention_events', '미수집')}`.",
                  f"- Decode attention profiler events: `{row.get('decode_attention_events', '미수집')}`.", ""]
        if row.get("status") == "validation_failed":
            lines += ["이 조건은 수치 검증에서 중단되어 warmup·free-generation 검증·지연 측정·profiler 수집을 실행하지 않았다.", ""]
    lines += ["수치 기준은 atol=0.02, rtol=0.02이며, reference는 compile을 끈 native "
              "Qwen3 + DynamicCache다. 같은 토큰으로 prefill 마지막 위치 및 매 decode 단계의 "
              "전체 vocabulary logits를 비교한다. K/V는 prefill과 step 1/4/32/64/127에서 "
              "모든 레이어·전체 유효 길이를 확인한다. 추가로 free-generation greedy token "
              "일치와 bulk/streaming 출력 일치를 검사한다. 정확도 검증은 성능 측정 밖이다.", "",
              "## 실행·정리", "",
              f"- 세션: `{lifecycle.get('session', '미확인')}`.",
              f"- 이 실행기 경과 시간: {lifecycle.get('elapsed_seconds', '미확인')}초. "
              "기존 세션 재사용 시 최초 할당부터의 총 시간과 다르다.",
              f"- 종료 명령 exit code: {lifecycle.get('stop_exit_code', '미확인')}.",
              f"- 서버 활성 세션 없음: {lifecycle.get('server_has_no_active_sessions', '미확인')}.", "",
              "원자료: [measurements.json](measurements.json), [execution.json](execution.json), "
              "[hardware.json](hardware.json), [lifecycle.json](lifecycle.json), "
              "[benchmark.log](benchmark.log). Profiler 수집 단계에 도달한 경우에만 trace가 저장된다.", ""]
    return "\n".join(lines)


def render_diagnosis(output: Path) -> str:
    def read(name, default):
        path = output / name
        return json.loads(path.read_text()) if path.exists() else default

    data = read("diagnosis.json", {})
    hardware = read("hardware.json", {})
    lifecycle = read("lifecycle.json", {})
    execution = read("execution.json", {})
    workload = data.get("workload", {})
    lines = ["# A100 Qwen3 compile 수치 오차 분리 실험", "",
             f"상태: **{execution.get('status', lifecycle.get('status', 'unknown'))}**.", "",
             f"- 실제 GPU: {hardware.get('gpu', '미확인')}, {hardware.get('gpu_memory_gib', '미확인')} GiB.",
             f"- PyTorch {hardware.get('torch', '미확인')}, CUDA {hardware.get('cuda', '미확인')}, "
             f"Transformers {hardware.get('transformers', '미확인')}.",
             f"- Qwen/Qwen3-14B, revision `{execution.get('revision', '미확인')}`, BF16, B=1.",
             f"- 저장된 동일 입력 {workload.get('prompt_len', '미확인')}토큰, "
             f"prefill 1회 + decode {workload.get('output_tokens', 1)-1}회 fixed-token replay.",
             "- 모델은 Transformers의 공식 Qwen3 구현이다. 모델 구조나 가중치는 교체하지 않았다.",
             "- 성능 비교가 아닌 정확도 원인 분리 실험이며, speculative decoding은 사용하지 않았다.",
             "- 모든 비교 기준: atol=0.02, rtol=0.02, NaN/Inf 실패. 기준을 완화하지 않았다.", "",
             "## 전체 모델 비교", "",
             "각 행은 왼쪽 실행을 오른쪽 reference와 비교한다. logits는 전체 vocabulary, "
             "K/V는 prefill과 decode 체크포인트의 모든 레이어·전체 유효 길이다.", "",
             "| 비교 | logits 실패/검사 | logits 최대 절대오차 | logits 최대 relative L2 | KV 실패/검사 | KV 최대 절대오차 | 단계별 argmax 일치 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, comparison in data.get("comparisons", {}).items():
        a, b = name.split("__vs__")
        logits, kv = comparison["logits"], comparison["kv"]
        lines.append(f"| {a} / {b} | {logits['failed_checks']}/{logits['checks']} | "
                     f"{logits['max_abs_error']:.6g} | {logits['max_relative_l2']:.6g} | "
                     f"{kv['failed_checks']}/{kv['checks']} | {kv['max_abs_error']:.6g} | "
                     f"{comparison['argmax_equal_all_steps']} |")
    lines += ["", "`default`는 기본 Inductor 수치 경로이고, `preserve_casts`는 "
              "`emulate_precision_casts=True`다. `auto`는 SDPA 자동 선택이며 `efficient`는 "
              "memory-efficient SDPA만 허용한 제어군이다. CUDA Graph 재생은 비활성화했다.", "",
              "## 실제 선택된 attention", ""]
    for name, case in data.get("cases", {}).items():
        lines += [f"### {name}", "", f"- 실행: {case.get('status', 'unknown')}.",
                  f"- Prefill events: `{case.get('prefill_attention_events', '미수집')}`.",
                  f"- Decode events: `{case.get('decode_attention_events', '미수집')}`.", ""]
    lines += ["## 첫 레이어 연산 분리", "",
              "각 연산은 동일 입력·가중치를 사용해 eager와 비교한다. 생성한 GPU 코드는 `micro-*.txt`에 보존했다.", "",
              "| 연산 | 기본 compile 최대 절대오차 | 기본 초과 원소 | cast 보존 최대 절대오차 | cast 보존 초과 원소 |",
              "|---|---:|---:|---:|---:|"]
    for name, test in data.get("microtests", {}).items():
        default, preserve = test["default"], test["preserve_casts"]
        lines.append(f"| {name} | {default['max_abs_error']:.6g} | {default['mismatches']} | "
                     f"{preserve['max_abs_error']:.6g} | {preserve['mismatches']} |")
    original_gate = data.get("comparisons", {}).get("compiled_static_preserve_casts__vs__eager_dynamic_auto", {})
    lines += ["", f"분리한 eager K 경로와 native 첫 레이어 K의 비교: "
              f"`{data.get('key_path_matches_native_eager_key', '미완료')}`.", "",
              f"cast 보존 설정의 원래 전체 logits/KV gate 통과: "
              f"**{data.get('preserve_casts_passes_original_gate', original_gate.get('passed', '미확인'))}**.", ""]
    if not data.get("microtests"):
        lines += ["연산 분리 microtest 단계에 도달하지 못했으므로 GPU micro 생성 코드는 수집하지 않았다.", ""]
    if execution.get("error"):
        lines += ["## 실행 오류", "", "```text", execution["error"], "```", ""]
    lines += ["## 실행과 정리", "",
              f"- 세션: `{lifecycle.get('session', '미확인')}`.",
              f"- 실행기 경과 시간: {lifecycle.get('elapsed_seconds', '미확인')}초.",
              f"- 종료 exit code: {lifecycle.get('stop_exit_code', '미확인')}.",
              f"- 서버 활성 세션 없음: {lifecycle.get('server_has_no_active_sessions', '미확인')}.", "",
              "원자료: [diagnosis.json](diagnosis.json), [hardware.json](hardware.json), "
              "[execution.json](execution.json), [benchmark.log](benchmark.log), [lifecycle.json](lifecycle.json).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_function = render_diagnosis if (args.output / "diagnosis.json").exists() else render
    (args.output / "report.md").write_text(render_function(args.output))


if __name__ == "__main__":
    main()
