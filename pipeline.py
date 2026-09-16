"""Run all three stages once, or on a fixed interval without overlapping runs."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock, Timeout

from penguins import ROOT

LOGGER = logging.getLogger("pipeline")


def run_once() -> int:
    started = time.monotonic()
    record = {"started_at": datetime.now(timezone.utc).isoformat()}
    LOGGER.info("Starting full pipeline: prepare -> train -> deploy")
    # Ensure DVC stage commands use the same interpreter, even without venv activation.
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MLFLOW_ENABLE_TELEMETRY"] = "false"
    log_path = ROOT / ".runtime/pipeline.log"
    with log_path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-m", "dvc", "repro", "--force", "--no-run-cache"],
                                   cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    record.update({"finished_at": datetime.now(timezone.utc).isoformat(),
                   "duration_seconds": round(time.monotonic() - started, 2), "exit_code": return_code})
    if return_code == 0:
        metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
        record["run_id"] = metadata["run_id"]
    with (ROOT / ".runtime/runs.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(record) + "\n")
    LOGGER.log(logging.INFO if return_code == 0 else logging.ERROR,
               "Full pipeline %s in %.2fs%s", "succeeded" if return_code == 0 else "failed",
               record["duration_seconds"], f"; run_id={record['run_id']}" if "run_id" in record else "")
    return return_code


def next_deadline(previous: float, now: float, interval: float) -> float:
    """Return the next future tick; never catch up by starting concurrent jobs."""
    following = previous + interval
    if following <= now:
        following += (int((now - following) // interval) + 1) * interval
    return following


def schedule(interval: int, max_runs: int | None = None, *, run=run_once,
             clock=time.monotonic, sleep=time.sleep) -> None:
    deadline = clock()
    count = 0
    while max_runs is None or count < max_runs:
        try:
            status = run()
            if status:
                LOGGER.error("Run failed; will retry at the next scheduled tick.")
        except Exception:
            LOGGER.exception("Run failed unexpectedly; will retry at the next scheduled tick.")
        count += 1
        if max_runs is not None and count >= max_runs:
            return
        now = clock()
        if now >= deadline + interval:
            LOGGER.warning("Run exceeded the interval; skipped missed ticks. Consider a longer interval.")
        deadline = next_deadline(deadline, now, interval)
        delay = max(0, deadline - clock())
        LOGGER.info("Next full pipeline run in %.1f seconds", delay)
        sleep(delay)


def main() -> int:
    # Windows redirected terminals can default to cp1252; MLflow emits Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run all stages once, then exit.")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-runs", type=int, help="Stop after N scheduled runs (useful for a demonstration).")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or (args.max_runs is not None and args.max_runs <= 0):
        parser.error("The interval and maximum run count must be positive.")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(runtime / "pipeline.log", encoding="utf-8")])
    try:
        with FileLock(runtime / "pipeline.lock", timeout=0):
            if args.once:
                return run_once()
            schedule(args.interval_seconds, args.max_runs)
    except Timeout:
        LOGGER.error("Another pipeline process is already running in this repository.")
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Scheduler stopped. Containers remain available; use docker compose down to stop them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
