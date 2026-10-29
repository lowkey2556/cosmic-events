"""Regressions found during independent handoff review; no astronomy network services needed."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_grid_is_selected_within_each_split_and_cache_tracks_inputs(tmp_path, monkeypatch):
    from cosmic import rganet

    out = tmp_path / "out"
    out.mkdir()
    cfg = SimpleNamespace(work_dir=str(tmp_path), output_dir=str(out))
    p = dict(grid=dict(lr=[0.0001, 0.0003], dropout=[0.1], hybrid_head=[False]), max_epochs=1)
    splits = pd.DataFrame([
        dict(seed=seed, auid=f"{seed}_{s}", set=s)
        for seed in (1, 2) for s in ("train", "val", "test")
    ])
    calls = []

    def train(data, split, variant, seed, params, device, **kwargs):
        assert set(split["seed"]) == {seed}
        assert set(split["set"]) == {"train", "val"}
        assert kwargs["evaluate_test"] is False
        calls.append(seed)
        # Different split -> different preferred setting. Cross-split reuse would fail this.
        score = params["lr"] if seed == 1 else -params["lr"]
        return None, None, dict(best_val_pr_auc=score, best_epoch=1, train_seconds=0.1)

    monkeypatch.setattr(rganet, "train_one", train)
    for seed, expected in ((1, 0.0003), (2, 0.0001)):
        best, _, grid = rganet.choose_config(cfg, {}, splits, p, None, {"full": {}}, seed, "data-a")
        assert best["lr"] == expected
        assert set(grid["grid_seed"]) == {seed}
    assert len(calls) == 4
    rganet.choose_config(cfg, {}, splits, p, None, {"full": {}}, 1, "data-a")
    assert len(calls) == 4  # unchanged inputs resume
    rganet.choose_config(cfg, {}, splits, p, None, {"full": {}}, 1, "data-b")
    assert len(calls) == 6  # changed data invalidates tuning
    changed = copy.deepcopy(p)
    changed["max_epochs"] = 2
    rganet.choose_config(cfg, {}, splits, changed, None, {"full": {}}, 1, "data-b")
    assert len(calls) == 8  # changed training configuration also invalidates tuning


def test_training_signature_uses_content_including_splits_and_vocab(tmp_path):
    from cosmic.rganet import training_signature

    cfg = SimpleNamespace(work_dir=str(tmp_path))
    names = ("windows_seq.npz", "ztf_windows_seq.npz", "splits.csv", "vocab.json")
    for name in names:
        (tmp_path / name).write_bytes(b"original")
    previous = training_signature(cfg)
    for name in names:
        path = tmp_path / name
        stat = path.stat()
        path.write_bytes(b"modified")  # same length, restored mtime: metadata is insufficient
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        current = training_signature(cfg)
        assert previous != current, name
        previous = current


def test_notebook_checks_out_main_after_review_branch_merge(tmp_path):
    import build_notebook

    notebook = build_notebook.build(str(tmp_path / "test.ipynb"))
    setup = next("".join(c["source"]) for c in notebook["cells"]
                 if c["cell_type"] == "code" and "REPO_URL =" in "".join(c["source"]))
    assert 'REPO_BRANCH = "main"' in setup
    assert "clone -q --branch {REPO_BRANCH}" in setup
    assert "switch {REPO_BRANCH}" in setup
    assert "merge --ff-only origin/{REPO_BRANCH}" in setup


def test_local_disk_setup_and_results_skip_drive_mount(tmp_path, monkeypatch):
    import ast
    import sys
    from types import ModuleType
    import build_notebook

    notebook = build_notebook.build(str(tmp_path / "test.ipynb"))
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    guards = []
    for source in sources:
        for node in ast.parse("\n".join(line for line in source.splitlines()
                                       if not line.startswith("!"))).body:
            if isinstance(node, ast.If) and "drive.mount(" in ast.unparse(node):
                guards.append(node)
    assert len(guards) == 2
    calls = []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=calls.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    for root, expected in (("/content/cosmic_roja", 0), ("/content/drive/MyDrive/cosmic_roja", 2)):
        calls.clear()
        env = dict(PROJECT_ROOT=root, V3=f"{root}/v3_5GB",
                   os=SimpleNamespace(path=SimpleNamespace(isdir=lambda path: False)))
        for guard in guards:
            exec(compile(ast.Module(body=[guard], type_ignores=[]), "mount-guard", "exec"), env)
        assert len(calls) == expected


def test_notebook_pipeline_stops_when_a_step_fails():
    import build_notebook

    calls = []

    def fail(cmd):
        calls.append(cmd)
        raise RuntimeError("step failed")

    with pytest.raises(RuntimeError, match="step failed"):
        exec(build_notebook.run("infer") + "\n" + build_notebook.run("compare"), {"sh": fail})
    assert calls == ["python run.py infer --data real"]


def test_clean_cache_changes_when_quality_cuts_change():
    from cosmic.config import load_config
    from cosmic.ztf_spark import _clean_key

    cfg = load_config("synthetic")
    man = pd.DataFrame({"md5": ["a" * 32, "b" * 32]})
    first = _clean_key(cfg, man)
    assert first == _clean_key(cfg, man.iloc[::-1])
    cfg.settings["ztf"]["saturation_mag"] += 0.1
    assert first != _clean_key(cfg, man)


def test_paired_comparison_counts_ties_and_missing_metrics():
    from cosmic.compare import paired

    rows = [dict(model=model, variant="v", level="window", seed=seed, mcc=value)
            for model, values in (("A", [0.5, 0.6, 0.3, np.nan]), ("B", [0.5, 0.4, 0.7, 0.2]))
            for seed, value in enumerate(values)]
    r = paired(pd.DataFrame(rows), ("A", "v"), ("B", "v"), "window", metrics=["mcc"]).iloc[0]
    assert (r.seeds, r.seeds_ahead, r.seeds_behind, r.seeds_tied) == (3, 1, 1, 1)


def test_rganet_run_tunes_two_seeds_and_resumes_only_matching_checkpoints(tmp_path, monkeypatch):
    import shutil
    import torch
    from cosmic import rganet
    from cosmic.config import load_config
    from test_pipeline import _toy_windows

    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    work = Path(cfg.work_dir)
    _toy_windows(work, n_stars=24, per_star=2, L=12)
    shutil.copyfile(work / "windows_seq.npz", work / "ztf_windows_seq.npz")
    cfg.settings["rganet"].update(d_model=8, heads=2, layers=1, n_time_freqs=2,
                                  pretrain_epochs=1, max_epochs=1, batch_size=64,
                                  grid=dict(lr=[0.0003], dropout=[0.1], hybrid_head=[False]))
    monkeypatch.setattr(rganet, "device_info", lambda: (torch.device("cpu"), "test CPU"))
    pred, _, runs = rganet.run(cfg, variants=["full", "no_pretrain"], seeds=[1, 2])
    grid = pd.read_csv(work / "rganet_grid.csv")
    assert set(grid.grid_seed) == {1, 2} and grid.chosen.all()
    assert len(runs) == 4 and pred.prob.between(0, 1).all()
    assert runs.checkpoint_sha256.str.len().eq(64).all()

    def should_not_train(*args, **kwargs):
        raise AssertionError("An unchanged completed run should resume")

    monkeypatch.setattr(rganet, "train_one", should_not_train)
    resumed, _, _ = rganet.run(cfg, variants=["full", "no_pretrain"], seeds=[1, 2])
    assert np.allclose(pred.prob, resumed.prob)
    # A separately overwritten model must not be paired with cached predictions from this run.
    (work / "models" / "rganet_full_seed1.pt").write_bytes(b"other model")
    with pytest.raises(AssertionError, match="unchanged completed run"):
        rganet.run(cfg, variants=["full", "no_pretrain"], seeds=[1, 2])
