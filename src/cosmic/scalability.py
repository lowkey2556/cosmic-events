"""Step 10 — Scalability test: Spark preprocessing + windowing through YARN at 1, 2 and 4 executors.

What it does: runs `python run.py preprocess+windows` as a fresh spark-submit to YARN for each
executor count (default 1, 2, 4), several times each (default 3), with the same executor size
every time (1 core, 1 GB). Each run reports the time of the two Spark steps (session start-up
excluded) and how many executors YARN actually granted. The table gives mean ± std time
and speedup = mean time at 1 executor / mean time at N executors, plus the machine's CPU
core count, so the result can be read honestly: on a 2-core Colab VM, more than 2 executors
cannot run truly in parallel.
Outputs: scalability_runs.csv (every run), scalability.csv (summary table; no chart).

Why: shows how the Big Data part of the pipeline behaves when YARN gives it more executors.
"""
import json
import os
import subprocess
import sys

import pandas as pd

from .config import REPO_ROOT


def cpu_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def run(cfg, overrides=()):
    s = cfg.section("scalability")
    out = cfg.output_dir
    # Fresh machine: build whatever preprocessing + windowing need (raw table, labels, observer scores)
    # once, BEFORE the timed runs, so no timed run includes that work.
    prep = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "run.py"), "preprocess+windows", "--data",
                           cfg.dataset, *sum((["--set", o] for o in overrides), []), "--only-missing"])
    if prep.returncode:
        raise RuntimeError("could not build the tables the scalability runs need")
    rows = []
    for n in s["executors"]:
        for r in range(1, int(s["runs"]) + 1):
            tj = os.path.join(cfg.work_dir, f"scal_{n}_{r}.json")
            cmd = [sys.executable, os.path.join(REPO_ROOT, "run.py"), "preprocess+windows", "--data", cfg.dataset,
                   *sum((["--set", o] for o in overrides), []),
                   "--set", f"spark.executor_instances={int(n)}",
                   "--set", f"spark.executor_cores={int(s['executor_cores'])}",
                   "--set", f"spark.executor_memory={s['executor_memory']}",
                   "--set", f"spark.executor_memory_overhead={s['executor_memory_overhead']}",
                   "--set", "windows.export_for_pytorch=false",
                   "--timings-json", tj]
            print(f"\n=== {n} executor(s), run {r}/{s['runs']}", flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:], res.stderr[-3000:])
                raise RuntimeError(f"scalability run failed ({n} executors, run {r})")
            with open(tj) as fh:
                t = json.load(fh)
            rows.append(dict(executors_requested=int(n), run=r, seconds=t["steps_seconds"],
                             executors_granted=t["executors_granted"], wall_seconds=t["wall_seconds"]))
            print(f"    {t['steps_seconds']:.1f} s (Spark steps), executors granted by YARN: {t['executors_granted']}")
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "scalability_runs.csv"), index=False)
    g = runs.groupby("executors_requested")
    summ = pd.DataFrame({"executors": g.size().index, "runs": g.size().to_numpy(),
                         "time_mean": g["seconds"].mean().to_numpy(), "time_std": g["seconds"].std(ddof=1).to_numpy(),
                         "executors_granted_min": g["executors_granted"].min().to_numpy()})
    base = summ.loc[summ["executors"] == summ["executors"].min(), "time_mean"].iloc[0]
    summ["speedup"] = base / summ["time_mean"]
    cores = cpu_cores()
    summ["machine_cpu_cores"] = cores
    summ.to_csv(os.path.join(out, "scalability.csv"), index=False)
    print(f"\nMachine CPU cores: {cores}\n" + summ.round(2).to_string(index=False))
    short = summ[summ["executors_granted_min"] < summ["executors"]]
    if len(short):
        print("NOTE: YARN granted fewer executors than requested for: "
              + ", ".join(f"{int(r.executors)} (got {int(r.executors_granted_min)})" for r in short.itertuples()))
    return summ
