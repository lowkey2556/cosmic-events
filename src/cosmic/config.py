"""Configuration loading and the project's paths.

What it does: reads config/settings.yaml, applies any `--set key.sub=value` overrides from the
command line, and resolves every path from ONE setting, the project root (PROJECT_ROOT):

    PROJECT_ROOT/data/            the AAVSO CSVs (only CSVs directly in this folder are read)
    PROJECT_ROOT/outputs/review2/ every Review 2 table and figure
    PROJECT_ROOT/work/            models, window tensors, predictions, labels/vsx_labels.csv and
                                  cache/ (the Hadoop/Spark/Hive tarballs)

PROJECT_ROOT is `datasets.real.project_root` in settings.yaml; the environment variable
COSMIC_PROJECT_ROOT overrides it (the notebook sets it). `check_paths` refuses any path inside
the retired folders (MyDrive/cosmic and cosmic_OLD_do_not_use), so they are never read or written.

Why: one root means the data, outputs and work files cannot drift apart. Real and synthetic
data are kept apart by construction (different HDFS folder, Hive database and output folder),
so synthetic results can never leak into the Review 2 deliverables.
"""
import copy
import os
from dataclasses import dataclass

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETIRED = ("/content/drive/MyDrive/cosmic", "cosmic_OLD_do_not_use")  # never read or written


def _parse_value(text):
    return yaml.safe_load(text)


def apply_overrides(settings, overrides):
    """Apply ['a.b=1', ...] overrides to a nested dict (returns a copy)."""
    out = copy.deepcopy(settings)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return out


def is_retired(path):
    """True if `path` is, or is inside, one of the retired folders."""
    p = os.path.splitdrive(os.path.abspath(str(path)))[1].replace("\\", "/").rstrip("/") + "/"
    old, renamed = RETIRED
    return (old + "/") in p or renamed in p.split("/")


@dataclass
class Config:
    settings: dict
    dataset: str

    @property
    def ds(self):
        return self.settings["datasets"][self.dataset]

    def _abs(self, path):
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    def _under_root(self, key):
        """A path from datasets.<name>.<key> if given (relative to the repo), else PROJECT_ROOT/paths.<key>."""
        own = self.ds.get(key)
        return self._abs(own) if own else os.path.join(self.root, self.settings["paths"][key])

    @property
    def root(self):
        """PROJECT_ROOT: $COSMIC_PROJECT_ROOT (real data only), else datasets.<name>.project_root."""
        env = os.environ.get("COSMIC_PROJECT_ROOT") if self.dataset == "real" else None
        return self._abs(env or self.ds["project_root"])

    @property
    def data_dir(self):
        """PROJECT_ROOT/data: the only folder read for AAVSO CSVs (not recursive)."""
        return self._under_root("data_subdir")

    raw_dir = data_dir  # older name, kept for callers

    @property
    def work_dir(self):
        path = self._under_root("work_subdir")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def labels_file(self):
        return self._under_root("labels_file")

    @property
    def cache_dir(self):
        """Tarball cache for setup/install_stack.sh."""
        return self._under_root("cache_subdir")

    @property
    def output_dir(self):
        path = self._under_root("output_subdir")
        if self.dataset != "real" and "review2" in path.replace("\\", "/").split("/"):
            raise ValueError("Synthetic outputs must never be written to review2/")
        os.makedirs(path, exist_ok=True)
        return path

    def paths(self):
        """Every resolved path, without creating anything."""
        return {"PROJECT_ROOT": self.root, "data (CSVs)": self.data_dir,
                "outputs": self._under_root("output_subdir"), "work": self._under_root("work_subdir"),
                "VSX labels file": self.labels_file, "tarball cache": self.cache_dir}

    @property
    def hdfs_base(self):
        return self.ds["hdfs_base"]

    def hdfs_uri(self, path):
        return self.settings["hdfs"]["namenode"] + path

    @property
    def db(self):
        return self.ds["database"]

    def table(self, name):
        return f"{self.db}.{name}"

    def section(self, name):
        return self.settings[name]

    def repo_path(self, rel):
        return os.path.join(REPO_ROOT, rel)

    def load_yaml(self, key):
        with open(self.repo_path(self.settings["paths"][key]), encoding="utf-8") as fh:
            return yaml.safe_load(fh)


def check_paths(cfg, need_data=True):
    """Print the resolved paths; raise a clear error if any points into a retired folder, or if the
    data folder is missing. Returns the paths."""
    paths = cfg.paths()
    width = max(len(k) for k in paths)
    for k, v in paths.items():
        print(f"  {k:{width}s}  {v}")
    bad = {k: v for k, v in paths.items() if is_retired(v)}
    if bad:
        raise ValueError("These paths point at the retired folder (MyDrive/cosmic or cosmic_OLD_do_not_use), "
                         f"which must never be used: {bad}. Set PROJECT_ROOT to /content/drive/MyDrive/cosmic_roja.")
    if need_data and not os.path.isdir(paths["data (CSVs)"]):
        raise FileNotFoundError(f"The data folder does not exist: {paths['data (CSVs)']}\n"
                                "Create it and put the AAVSO CSV files directly inside it.")
    return paths


def load_config(dataset="real", overrides=None, path=None):
    path = path or os.path.join(REPO_ROOT, "config", "settings.yaml")
    with open(path, encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    settings = apply_overrides(settings, overrides)
    if dataset not in settings["datasets"]:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return Config(settings=settings, dataset=dataset)
