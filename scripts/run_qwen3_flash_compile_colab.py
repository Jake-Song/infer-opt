#!/usr/bin/env python3
"""Run one bounded A100 measurement and always recover/stop its session."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import textwrap
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", help="Reuse this task's already-owned session; stop it after this run")
    parser.add_argument("--attention", choices=["flash", "auto"], default="flash")
    parser.add_argument("--gpu-memory-gb", type=int, choices=[40, 80], default=80)
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--official-defaults", action="store_true", help="Use native generate and default CompileConfig")
    parser.add_argument("--fixture", type=Path, help="Saved measurements.json containing prompt and fixed replay tokens")
    parser.add_argument("--budget-minutes", type=float, default=120)
    args = parser.parse_args()
    if args.budget_minutes < 15 or args.budget_minutes > 120:
        parser.error("budget must be between 15 and 120 minutes")
    if args.diagnose and (args.fixture is None or not args.fixture.is_file() or args.attention != "auto"):
        parser.error("diagnosis requires --attention auto and an existing --fixture measurements.json")
    if args.official_defaults and (args.diagnose or args.emulate_precision_casts or args.attention != "auto"):
        parser.error("official defaults requires --attention auto, no diagnosis/cast overrides")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "qwen3-flash-" + datetime.now(timezone.utc).strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    session = args.session or run_id
    started = time.monotonic()
    lifecycle = {"session": session, "budget_minutes": args.budget_minutes,
                 "requested_gpu": f"A100 {args.gpu_memory_gb}GB", "status": "starting",
                 "run_id": run_id, "reused_session": bool(args.session), "attention": args.attention,
                 "emulate_precision_casts": args.emulate_precision_casts, "official_defaults": args.official_defaults}
    source = Path(__file__).with_name("measure_qwen3_generate_defaults.py" if args.official_defaults else ("diagnose_qwen3_compile.py" if args.diagnose else "measure_qwen3_flash_compile.py"))
    shutil.copyfile(source, output / source.name)
    lifecycle["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(__file__, output / Path(__file__).name)

    def save():
        (output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

    def command(arguments, timeout=120, check=True, capture=False):
        print(" ".join(arguments), flush=True)
        with (output / "colab.log").open("a") as log:
            log.write("\n" + " ".join(arguments) + "\n")
            log.flush()
            result = subprocess.run(arguments, stdout=subprocess.PIPE if capture else log,
                                    stderr=subprocess.STDOUT, text=True, timeout=timeout)
            if capture:
                log.write(result.stdout)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {arguments[:2]}; see colab.log")
        return result

    attempted_allocation = False
    try:
        command(["colab", "version"])
        help_text = command(["colab", "new", "--help"], capture=True).stdout
        if args.gpu_memory_gb == 80 and "--high-mem" not in help_text:
            raise RuntimeError("Installed CLI lacks --high-mem")
        listing = command(["colab", "sessions"], capture=True).stdout
        if args.session and f"[{session}]" not in listing:
            raise RuntimeError("Requested owned session is not active")
        if not args.session and "No active sessions found on server." not in listing:
            raise RuntimeError("Active or unrecognized sessions; refusing duplicate allocation")
        lifecycle["stage"] = "allocation"
        lifecycle["session_start_utc"] = datetime.now(timezone.utc).isoformat()
        if not args.session:
            lifecycle["allocation_attempt_utc"] = lifecycle["session_start_utc"]
        save()
        attempted_allocation = True
        if not args.session:
            allocation = ["colab", "new", "-s", session, "--gpu", "A100"]
            if args.gpu_memory_gb == 80:
                allocation.append("--high-mem")
            command(allocation, timeout=300)
        lifecycle["stage"] = "execution"
        save()
        command(["colab", "upload", "-s", session, str(source), "content/" + run_id + ".py"])
        worker_options = ['--attention', args.attention, '--gpu-memory-gb', str(args.gpu_memory_gb)]
        if args.emulate_precision_casts:
            worker_options.append('--emulate-precision-casts')
        if args.diagnose:
            helper = Path(__file__).with_name("measure_qwen3_flash_compile.py")
            shutil.copyfile(helper, output / helper.name)
            shutil.copyfile(args.fixture, output / "replay_fixture.json")
            command(["colab", "upload", "-s", session, str(helper), "content/" + run_id + "-helper.py"])
            command(["colab", "upload", "-s", session, str(args.fixture), "content/" + run_id + "-fixture.json"])
            worker_options += ['--helper', '/content/' + run_id + '-helper.py',
                               '--fixture', '/content/' + run_id + '-fixture.json']
        remaining = int(args.budget_minutes * 60 - (time.monotonic() - started) - 600)
        if remaining < 60:
            raise TimeoutError("No time remains after reserving ten minutes for recovery and cleanup")
        code = textwrap.dedent(f"""\
            import json, os, subprocess, sys, time, traceback, zipfile
            from pathlib import Path
            import torch
            root = Path('/content') / {run_id!r}
            root.mkdir(exist_ok=True)
            (root / 'allocation_hardware.json').write_text(json.dumps({{
                'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
                'memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
            }}, indent=2))
            deadline = time.monotonic() + {remaining}
            status = {{'status': 'running'}}
            try:
                memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
                if 'A100' not in torch.cuda.get_device_name() or memory_gib < {35 if args.gpu_memory_gb == 40 else 75}:
                    raise RuntimeError('Requested A100 {args.gpu_memory_gb}GB not assigned')
                if {args.gpu_memory_gb == 40!r} and memory_gib >= 48:
                    raise RuntimeError('Requested 40GB but received a different memory class')
                with (root / 'install.log').open('w') as log:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                                    'transformers==4.57.3', 'accelerate>=1,<2', 'huggingface_hub<1'],
                                   stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=min(600, max(1, deadline-time.monotonic())))
                env = dict(os.environ, TORCH_LOGS='recompiles', TOKENIZERS_PARALLELISM='false',
                           TORCHINDUCTOR_COMPILE_THREADS='4')
                with (root / 'benchmark.log').open('w') as log:
                    result = subprocess.run([sys.executable, '/content/' + {run_id!r} + '.py',
                                             '--output', str(root)] + {worker_options!r}, stdout=log, stderr=subprocess.STDOUT,
                                            env=env, timeout=max(1, deadline-time.monotonic()))
                status = {{'status': 'finished', 'returncode': result.returncode}}
            except Exception as exc:
                status = {{'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}}
                (root / 'supervisor_error.txt').write_text(traceback.format_exc())
            finally:
                (root / 'supervisor.json').write_text(json.dumps(status, indent=2))
                with zipfile.ZipFile('/content/' + {run_id!r} + '.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                    for path in root.rglob('*'):
                        if path.is_file() and path.suffix != ".pt":
                            archive.write(path, str(path.relative_to(root)))
                (root / 'done.json').write_text(json.dumps(status))
                print(json.dumps(status), flush=True)
                print((root / 'benchmark.log').read_text()[-12000:] if (root / 'benchmark.log').exists() else '')
            """)
        remote = output / "remote_execution.py"
        remote.write_text(code)
        # Run the supervisor independently from the notebook websocket. A lost
        # client connection cannot restart the benchmark or discard its files.
        command(["colab", "upload", "-s", session, str(remote), "content/" + run_id + "-supervisor.py"])
        launch = output / "remote_launch.py"
        launch.write_text(textwrap.dedent(f"""\
            import json, subprocess, sys
            from pathlib import Path
            root = Path('/content') / {run_id!r}
            root.mkdir(exist_ok=True)
            marker = root / 'launched.json'
            if not marker.exists():
                log = (root / 'supervisor.log').open('a')
                process = subprocess.Popen([sys.executable, '/content/' + {run_id!r} + '-supervisor.py'],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                marker.write_text(json.dumps({{'pid': process.pid}}))
            print(marker.read_text(), flush=True)
            """))
        try:
            command(["colab", "exec", "-s", session, "-f", str(launch), "--timeout", "60"],
                    timeout=90, check=False)
        except subprocess.TimeoutExpired:
            lifecycle["launch_client_timeout"] = True
        polling_deadline = time.monotonic() + remaining + 30
        status_file = output / "supervisor.json"
        while time.monotonic() < polling_deadline:
            result = command(["colab", "download", "-s", session,
                              "content/" + run_id + "/done.json", str(status_file)],
                             timeout=60, check=False, capture=True)
            if result.returncode == 0:
                break
            progress = output / "progress.json"
            result = command(["colab", "download", "-s", session,
                              "content/" + run_id + "/progress.json", str(progress)],
                             timeout=60, check=False, capture=True)
            if result.returncode == 0:
                print(progress.read_text(), flush=True)
            time.sleep(25)
        else:
            raise TimeoutError("Remote supervisor did not finish within budget")
        lifecycle["stage"] = "recovery"
        save()
        archive_path = output / "remote_results.zip"
        command(["colab", "download", "-s", session, "content/" + run_id + ".zip", str(archive_path)], timeout=180)
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe archive entry")
            archive.extractall(output)
        lifecycle["status"] = json.loads((output / "execution.json").read_text())["status"]
    except Exception as exc:
        lifecycle["status"] = "failed"
        lifecycle["error"] = f"{type(exc).__name__}: {exc}"
        print(lifecycle["error"], flush=True)
    finally:
        if attempted_allocation:
            try:
                lifecycle["stop_exit_code"] = command(["colab", "stop", "-s", session], timeout=120, check=False).returncode
            except Exception as exc:
                lifecycle["cleanup_error"] = str(exc)
        try:
            listing = command(["colab", "sessions"], capture=True, check=False)
            lifecycle["final_sessions_output"] = listing.stdout
            lifecycle["server_has_no_active_sessions"] = (
                listing.returncode == 0 and "No active sessions found on server." in listing.stdout)
        except Exception as exc:
            lifecycle["sessions_error"] = str(exc)
        lifecycle["elapsed_seconds"] = time.monotonic() - started
        save()
    print(json.dumps(lifecycle, indent=2), flush=True)
    return 0 if lifecycle["status"] in ("passed", "diagnosed") and lifecycle.get("server_has_no_active_sessions") else 1


if __name__ == "__main__":
    raise SystemExit(main())
