"""Test orchestration failure/cleanup without contacting Colab or allocating GPUs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_colab.py"
    spec = importlib.util.spec_from_file_location("colab_baseline_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allocation_failure_does_not_retry_and_checks_cleanup(monkeypatch, tmp_path):
    module = load_runner()
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        message = "No active sessions found on server.\n" if args[1] == "sessions" else "test response\n"
        if kwargs["stdout"] != subprocess.PIPE:
            kwargs["stdout"].write(message)
        return subprocess.CompletedProcess(args, 1 if args[1] == "new" else 0, stdout=message)

    def bundle(destination):
        destination.write_bytes(b"test source")
        return {"sha256": "test"}

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "build_bundle", bundle)
    output = tmp_path / "failed-run"
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(output)])
    assert module.main() == 1
    assert len([c for c in commands if c[1] == "new"]) == 1
    assert commands[-2][1] == "stop"
    assert commands[-1][1] == "sessions"
    assert not any(c[1] in ("exec", "upload") for c in commands)
    lifecycle = json.loads((output / "lifecycle.json").read_text())
    assert lifecycle["status"] == "error"
    assert lifecycle["server_has_no_active_sessions"]
    assert (output / "report.md").exists()


def test_remote_launcher_is_valid_python_and_preserves_cleanup_artifacts():
    module = load_runner()
    code = module.remote_code("test-session", "content/source.tar.gz", 100)
    compile(code, "remote_execution.py", "exec")


@pytest.mark.parametrize("stop_session", [False, True])
@pytest.mark.parametrize("upload_fails", [False, True])
def test_existing_session_never_allocates_and_only_stops_when_requested(
    monkeypatch, tmp_path, stop_session, upload_fails,
):
    module = load_runner()
    commands = []
    session = "manual-a100"
    listing = f"[{session}] endpoint | Hardware: A100 | Shape: STANDARD | Variant: GPU\n"

    def run(args, **kwargs):
        commands.append(args)
        stopped = any(c[1] == "stop" for c in commands)
        message = ("No active sessions found on server.\n" if stopped else listing)
        if kwargs["stdout"] != subprocess.PIPE:
            kwargs["stdout"].write(message)
        if args[1] == "download":
            with zipfile.ZipFile(args[-1], "w") as archive:
                archive.writestr("execution.json", json.dumps({"status": "passed"}))
                archive.writestr("report.md", "passed\n")
        return subprocess.CompletedProcess(
            args, 1 if upload_fails and args[1] == "upload" else 0, stdout=message,
        )

    def bundle(destination):
        destination.write_bytes(b"test source")
        return {"sha256": "test"}

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "build_bundle", bundle)
    output = tmp_path / "reuse"
    arguments = ["runner", "--session", session, "--output", str(output)]
    if stop_session:
        arguments.append("--stop-session")
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == (1 if upload_fails else 0)
    assert not any(c[1] == "new" for c in commands)
    assert [c for c in commands if c[1] == "stop"] == (
        [["colab", "stop", "-s", session]] if stop_session else []
    )
    lifecycle = json.loads((output / "lifecycle.json").read_text())
    assert lifecycle["session_reused"]
    assert lifecycle["target_session_active"] is not stop_session
    if not upload_fails:
        code = (output / "remote_execution.py").read_text()
        compile(code, "remote_execution.py", "exec")
        # Reusing a session must not overwrite a previous run's remote results.
        assert lifecycle["run_id"] != session
        assert repr(lifecycle["run_id"]) in code


@pytest.mark.parametrize("requested_session", [None, "missing-session"])
def test_existing_sessions_require_explicit_valid_selection(monkeypatch, tmp_path, requested_session):
    module = load_runner()
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        message = "[manual-a100] endpoint | Hardware: A100 | Shape: STANDARD | Variant: GPU\n"
        if kwargs["stdout"] != subprocess.PIPE:
            kwargs["stdout"].write(message)
        return subprocess.CompletedProcess(args, 0, stdout=message)

    monkeypatch.setattr(module.subprocess, "run", run)
    arguments = ["runner", "--output", str(tmp_path / "result")]
    if requested_session:
        arguments += ["--session", requested_session]
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == 1
    assert not any(c[1] in ("new", "stop", "upload", "exec") for c in commands)
