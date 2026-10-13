#!/usr/bin/env python3
"""Run the baseline on a new A100 or an explicitly selected existing session.

Run from the project Python environment. Colab CLI must already be authenticated.
No Drive mount or changes to the runtime's CUDA PyTorch installation are needed.
New sessions are stopped after execution. Existing sessions are preserved unless
--stop-session is supplied.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infer_opt.baseline import source_identity


def active_session_names(listing: str) -> set[str]:
    """Parse the installed CLI's session listing without reading credential files."""
    if "No active sessions found on server." in listing:
        return set()
    names = set(re.findall(r"^\[([^\]]+)\] .+ \| Hardware:", listing, re.MULTILINE))
    if not names:
        raise RuntimeError("Could not identify active sessions; inspect colab.log before allocating")
    return names


def build_bundle(destination: Path) -> dict:
    source = source_identity()
    with tarfile.open(destination, "w:gz") as archive:
        paths = [ROOT / name for name in source["files"]]
        paths += sorted((ROOT / "tests").glob("test_*.py"))
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        data = json.dumps(source, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("source_manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return source


def remote_code(run_id: str, remote_archive: str, budget_seconds: int) -> str:
    return textwrap.dedent(f"""\
        import importlib.util
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tarfile
        import time
        import traceback
        import zipfile

        started = time.monotonic()
        root = Path('/content') / {run_id!r}
        root.mkdir(exist_ok=True)
        with tarfile.open('/' + {remote_archive!r}) as archive:
            archive.extractall(root, filter='data')
        output = root / 'results'
        output.mkdir(exist_ok=True)
        os.chdir(root)
        sys.path.insert(0, str(root))
        status = {{'status': 'running', 'requested_gpu': 'A100'}}
        try:
            import torch
            from infer_opt.baseline import environment
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable on requested A100 runtime')
            hardware = environment(torch.device('cuda'))
            (output / 'hardware.json').write_text(json.dumps(hardware, indent=2))
            print(json.dumps(hardware, indent=2), flush=True)
            if 'A100' not in hardware['gpu_name']:
                raise RuntimeError('Requested A100 but received ' + hardware['gpu_name'])
            if not hardware['bf16_supported']:
                raise RuntimeError('BF16 unsupported')
            if importlib.util.find_spec('pytest') is None:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest>=8'], check=True, timeout=120)
            test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root,
                                  capture_output=True, text=True, timeout=300)
            (output / 'tests.txt').write_text(test.stdout + test.stderr)
            print(test.stdout + test.stderr, flush=True)
            if test.returncode:
                raise RuntimeError('Remote regression tests failed')
            remaining = {budget_seconds} - (time.monotonic() - started)
            if remaining < 60:
                raise TimeoutError('Budget exhausted before measurement')
            command = [sys.executable, '-m', 'infer_opt', 'baseline', '--output', str(output),
                       '--device', 'cuda', '--expected-gpu', 'A100', '--budget-minutes', str(remaining / 60)]
            with (output / 'benchmark.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=remaining, cwd=root)
            print((output / 'benchmark.log').read_text(), flush=True)
            if completed.returncode:
                raise RuntimeError('Baseline exited with status ' + str(completed.returncode))
            status['status'] = 'passed'
        except Exception as exc:
            status['status'] = 'error'
            status['error'] = type(exc).__name__ + ': ' + str(exc)
            traceback.print_exc()
        finally:
            status['elapsed_seconds'] = time.monotonic() - started
            (output / 'execution.json').write_text(json.dumps(status, indent=2))
            with zipfile.ZipFile('/content/' + {run_id!r} + '-results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output.rglob('*')):
                    if path.is_file():
                        archive.write(path, str(path.relative_to(output)))
            print('BASELINE_EXECUTION=' + json.dumps(status), flush=True)
    """)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=120)
    parser.add_argument("--session", help="Reuse this active Colab session instead of allocating a new one")
    parser.add_argument("--stop-session", action="store_true",
                        help="Also stop a reused session after the run (new sessions are always stopped)")
    args = parser.parse_args()
    if args.budget_minutes < 10:
        parser.error("allow at least 10 minutes for allocation, recovery and cleanup")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "baseline-a100-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    session = args.session or run_id
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    lifecycle = {"session": session, "requested_gpu": "A100", "status": "starting",
                 "budget_minutes": args.budget_minutes, "run_id": run_id,
                 "session_reused": bool(args.session), "colab_executable": shutil.which("colab")}

    def stage(name: str) -> None:
        lifecycle["stage"] = name
        (output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

    def command(arguments: list[str], *, timeout: float = 120, input_text: str | None = None,
                check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        with (output / "colab.log").open("a") as log:
            log.write("\n" + " ".join(arguments) + "\n")
            log.flush()
            completed = subprocess.run(arguments, input=input_text, text=True,
                                       stdout=subprocess.PIPE if capture else log,
                                       stderr=subprocess.STDOUT, timeout=timeout)
            if capture:
                log.write(completed.stdout)
        if check and completed.returncode:
            raise RuntimeError(f"{arguments[:2]} exited with status {completed.returncode}; see colab.log")
        return completed

    should_stop = False
    try:
        stage("preflight")
        command(["colab", "version"])
        command(["colab", "help", "new"])
        sessions = command(["colab", "sessions"], capture=True)
        active = active_session_names(sessions.stdout)
        if args.session:
            if session not in active or session == "?":
                raise RuntimeError(f"Session {session!r} is not active; inspect `colab sessions`")
        elif active:
            raise RuntimeError("Active Colab sessions exist; select one with --session instead of allocating another")
        with tempfile.TemporaryDirectory(prefix="infer-opt-baseline-") as temporary:
            bundle = Path(temporary) / "source.tar.gz"
            source = build_bundle(bundle)
            (output / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
            # Retain the exact source snapshot, including uncommitted benchmark code.
            (output / "source.tar.gz").write_bytes(bundle.read_bytes())
            if args.session:
                print(f"Reusing {session}; no new allocation", flush=True)
                should_stop = args.stop_session
                lifecycle["status"] = "reused"
            else:
                stage("allocation")
                should_stop = True
                print(f"Allocating {session} (A100)", flush=True)
                command(["colab", "new", "-s", session, "--gpu", "A100"], timeout=300)
                lifecycle["status"] = "allocated"
            stage("upload")
            remote_archive = "content/" + run_id + "-source.tar.gz"
            command(["colab", "upload", "-s", session, str(bundle), remote_archive])
            remaining = int(deadline - time.monotonic() - 300)
            if remaining < 60:
                raise TimeoutError("budget exhausted before remote execution")
            # Each run gets a fresh remote directory, even on a reused session.
            code = remote_code(run_id, remote_archive, remaining)
            (output / "remote_execution.py").write_text(code)
            print("Running hardware verification, tests and four baseline conditions", flush=True)
            stage("execution")
            command(["colab", "exec", "-s", session, "--timeout", str(remaining + 30)],
                    timeout=remaining + 60, input_text=code, check=False)
            stage("recovery")
            archive_path = Path(temporary) / "results.zip"
            command(["colab", "download", "-s", session, "content/" + run_id + "-results.zip", str(archive_path)])
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    if Path(name).name != name:
                        raise ValueError("unexpected nested result path")
                    (output / name).write_bytes(archive.read(name))
            execution = json.loads((output / "execution.json").read_text())
            lifecycle["status"] = execution["status"]
            if execution["status"] != "passed":
                lifecycle["error"] = execution.get("error", "remote failure")
    except Exception as exc:
        lifecycle["status"] = "error"
        lifecycle["error"] = f"{type(exc).__name__}: {exc}"
        print(lifecycle["error"], flush=True)
    finally:
        if should_stop:
            try:
                stopped = command(["colab", "stop", "-s", session], check=False)
                lifecycle["stop_exit_code"] = stopped.returncode
            except Exception as exc:
                lifecycle["cleanup_error"] = str(exc)
        try:
            synced = command(["colab", "sessions"], check=False, capture=True)
            lifecycle["final_sessions_exit_code"] = synced.returncode
            lifecycle["final_sessions_output"] = synced.stdout.strip()
            lifecycle["server_has_no_active_sessions"] = (
                synced.returncode == 0 and "No active sessions found on server." in synced.stdout
            )
            if synced.returncode == 0:
                lifecycle["target_session_active"] = session in active_session_names(synced.stdout)
        except Exception as exc:
            lifecycle["sessions_error"] = str(exc)
        lifecycle["elapsed_seconds"] = time.monotonic() - started
        (output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")
        if not (output / "report.md").exists():
            (output / "report.md").write_text(
                "# A100 베이스라인 실행 기록\n\n"
                f"상태: **{lifecycle['status']}**. GPU 성능 측정 결과는 생성되지 않았다.\n\n"
                f"오류: {lifecycle.get('error', 'unknown')}\n\n"
                f"요청 환경: Colab A100. 세션: `{session}`.\n\n"
                f"실패 단계: `{lifecycle.get('stage', 'unknown')}`. "
                "명령 출력은 `colab.log`, 실행 및 정리 기록은 `lifecycle.json`에 있다.\n\n"
                "서버 상태:\n\n```text\n" + lifecycle.get("final_sessions_output", "확인 실패") + "\n```\n"
            )
    print(json.dumps(lifecycle, indent=2), flush=True)
    cleanup_ok = not should_stop or (
        lifecycle.get("stop_exit_code") == 0 and lifecycle.get("target_session_active") is False
    )
    return 0 if lifecycle["status"] == "passed" and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
