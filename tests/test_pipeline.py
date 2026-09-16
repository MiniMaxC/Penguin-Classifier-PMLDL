import os
from pathlib import Path
import shutil
import subprocess
import sys

from conftest import ROOT
from pipeline import next_deadline, schedule


def test_next_tick_skips_overruns():
    assert next_deadline(0, 20, 300) == 300
    assert next_deadline(0, 650, 300) == 900
    assert next_deadline(0, 300, 300) == 600


def test_scheduler_retries_failures_and_never_overlaps():
    now = [0.0]
    starts = []
    def run():
        starts.append(now[0])
        now[0] += 650 if len(starts) == 1 else 10
        return 1 if len(starts) == 1 else 0
    def sleep(seconds):
        now[0] += seconds
    schedule(300, max_runs=3, run=run, clock=lambda: now[0], sleep=sleep)
    assert starts == [0.0, 900.0, 1200.0]


def test_training_failure_stops_dvc_before_deployment(tmp_path):
    # Exercise the real stage graph in an isolated project; no Docker operations.
    for folder in ["code", "data/raw"]:
        shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["dvc.yaml", "penguins.py", "requirements.txt", "requirements-model.txt", ".dockerignore"]:
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "code/models/train.py").write_text("raise RuntimeError('intentional training failure')\n")
    (tmp_path / "code/deployment/deploy.py").write_text(
        "from pathlib import Path\nPath('deployment_was_called').write_text('bad')\n")
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
           "DVC_NO_ANALYTICS": "true"}
    subprocess.run([sys.executable, "-m", "dvc", "init", "--no-scm", "--quiet"], cwd=tmp_path, env=env, check=True)
    result = subprocess.run([sys.executable, "-m", "dvc", "repro", "--force", "--no-run-cache"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "intentional training failure" in result.stdout + result.stderr
    assert (tmp_path / "data/processed/train.csv").exists()
    assert not (tmp_path / "deployment_was_called").exists()
