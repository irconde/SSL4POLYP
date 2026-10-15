#!/usr/bin/env python3
"""Run one phase of the campaign v3 matrix on the local GPUs (one run per GPU).

Phases: ``sun_full``, ``e2e3`` (E2 and E3 evaluations of the sun_full models), ``e1``, ``e4``.
The run list and command lines come from ``campaign_plan.run_matrix`` (the dry-run matrix).

* Resumable: a run is complete when its output directory holds ``RUN_COMPLETE.json``. A directory
  without it (interrupted run) is moved to ``checkpoints_v3/campaign/_aborted/`` and the run starts
  from scratch; runs are never resumed mid-training.
* Transient failures (out of memory, I/O, DataLoader worker crash) are retried once.
* Any other failure, or a second failure, stops the phase: running jobs finish, no new job starts.
* Events go to ``results_v3/campaign/<phase>/events.jsonl``; each attempt's output to ``logs/``.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_plan as cp  # noqa: E402

PHASES = {"sun_full": ("sun_full",), "e2e3": ("E2", "E3"), "e1": ("E1",), "e4": ("E4",)}
TRANSIENT = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|Input/output error|OSError|IOError"
    r"|DataLoader worker \(pid|Broken pipe|Too many open files|No space left",
)
ABORTED = REPO / "checkpoints_v3/campaign/_aborted"


def run_name(run: dict) -> str:
    variant = f"_{run['variant']}" if run["variant"] else ""
    return f"{run['experiment']}_k{run['k']}_{run['regime']}_seed{run['seed']}{variant}"


def output_dir(run: dict) -> Path:
    argv = run["argv"]
    return REPO / argv[argv.index("--output-dir") + 1]


def is_complete(run: dict) -> bool:
    return (output_dir(run) / "RUN_COMPLETE.json").exists()


def produced_metrics(run: dict) -> bool:
    out = output_dir(run)
    if run["experiment"] in ("E2", "E3"):
        return any(not p.name.endswith("_last.metrics.json") for p in out.rglob("*.metrics.json"))
    return any(out.rglob("*_last.metrics.json"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--gpus", default="0 1")
    args = parser.parse_args()
    os.chdir(REPO)

    import timm  # noqa: F401  (environment guard)

    if timm.__version__ != "0.4.12":
        sys.exit("timm 0.4.12 required: activate ssl4polyp-ft")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"Tracked files modified; commit before running the campaign:\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    phase_dir = REPO / "results_v3/campaign" / args.phase
    (phase_dir / "logs").mkdir(parents=True, exist_ok=True)
    events_path = phase_dir / "events.jsonl"
    lock = threading.Lock()

    def event(**payload) -> None:
        with lock, events_path.open("a") as handle:
            handle.write(json.dumps({"time": now(), "commit": commit, **payload}) + "\n")

    runs = [r for r in cp.run_matrix() if r["experiment"] in PHASES[args.phase]]
    todo = [r for r in runs if not is_complete(r)]
    event(kind="phase_start", phase=args.phase, expected=len(runs), already_complete=len(runs) - len(todo), gpus=args.gpus)
    print(f"[{now()}] phase {args.phase}: {len(runs)} runs, {len(runs) - len(todo)} already complete", flush=True)

    jobs: "queue.Queue[dict]" = queue.Queue()
    for r in todo:
        jobs.put(r)
    stop = threading.Event()
    failures: list = []

    def attempt(run: dict, gpu: str, n: int) -> tuple:
        name, out = run_name(run), output_dir(run)
        if out.exists():
            ABORTED.mkdir(parents=True, exist_ok=True)
            target = ABORTED / f"{name}_{datetime.now():%Y%m%dT%H%M%S}"
            shutil.move(str(out), str(target))
            event(kind="moved_incomplete", run=name, to=str(target.relative_to(REPO)))
        if "--parent-checkpoint" in run["argv"]:
            parent = REPO / run["argv"][run["argv"].index("--parent-checkpoint") + 1]
            if not parent.exists():
                return False, False, f"parent checkpoint missing: {parent.relative_to(REPO)}", 0.0, time.time()
        log = phase_dir / "logs" / f"{name}.attempt{n}.stdout"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        start = time.time()
        with log.open("w") as handle:
            rc = subprocess.run([sys.executable, "-m", "ssl4polyp.classification.train_classification", *run["argv"]],
                                stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
        elapsed = round(time.time() - start, 1)
        ok = rc == 0 and produced_metrics(run)
        text = log.read_text(errors="replace")[-20000:]
        transient = bool(TRANSIENT.search(text)) and not ok
        reason = "" if ok else (f"exit code {rc}" + ("; transient error in log" if transient else ""))
        if ok:
            (out / "RUN_COMPLETE.json").write_text(json.dumps(
                {"run": name, "gpu": gpu, "attempt": n, "start": datetime.fromtimestamp(start).astimezone().isoformat(timespec="seconds"),
                 "elapsed_s": elapsed, "commit": commit, "argv": run["argv"]}, indent=2))
        return ok, transient, reason, elapsed, start

    def worker(gpu: str) -> None:
        while not stop.is_set():
            try:
                run = jobs.get_nowait()
            except queue.Empty:
                return
            name = run_name(run)
            for n in (1, 2):
                ok, transient, reason, elapsed, start = attempt(run, gpu, n)
                event(kind="run", run=name, experiment=run["experiment"], gpu=gpu, attempt=n,
                      start=datetime.fromtimestamp(start).astimezone().isoformat(timespec="seconds"),
                      elapsed_s=elapsed, status="ok" if ok else "failed", reason=reason)
                print(f"[{now()}] {name} gpu={gpu} attempt={n} {'ok' if ok else 'FAILED ' + reason} {elapsed}s", flush=True)
                if ok:
                    break
                if not (transient and n == 1):
                    failures.append({"run": name, "reason": reason})
                    stop.set()
                    break

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    complete = sum(is_complete(r) for r in runs)
    event(kind="phase_end", phase=args.phase, expected=len(runs), complete=complete, failures=failures)
    print(f"[{now()}] phase {args.phase} end: {complete}/{len(runs)} complete, failures={failures}", flush=True)
    return 0 if complete == len(runs) and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
