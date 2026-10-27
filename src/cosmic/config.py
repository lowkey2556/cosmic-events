"""Configuration loading and the project's paths.

What it does: reads config/settings.yaml, applies any `--set key.sub=value` overrides from the
command line, and resolves every path from ONE setting, the project root (PROJECT_ROOT):

    PROJECT_ROOT/data/              AAVSO CSVs. Read only (only CSVs directly in this folder).
    PROJECT_ROOT/work/cache/        Hadoop/Spark/Hive/Derby tarballs. Read only, reused.
    PROJECT_ROOT/work/labels/       VSX labels (vsx_labels.csv), reused and extended for new stars.
    PROJECT_ROOT/v3_5GB/            V3_ROOT: EVERYTHING the v3 pipeline writes goes here:
        ztf_raw/  clean/  work/  outputs/  dashboard_data/

PROJECT_ROOT is `datasets.real.project_root` in settings.yaml (environment variable
COSMIC_PROJECT_ROOT overrides it); V3_ROOT is PROJECT_ROOT/v3_5GB (V3_ROOT overrides it).
`check_paths` refuses any path inside the retired folders (MyDrive/cosmic, cosmic_OLD_do_not_use,
_OLD_100MB), and any write path outside V3_ROOT, so old and new results can never mix.

Why: one root means the data, outputs and work files cannot drift apart. Real and synthetic
data are kept apart by construction (different HDFS folder, Hive database and output folder),
so synthetic results can never leak into the deliverables.
"""
import copy
import os
from dataclasses import dataclass

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETIRED = ("/content/drive/MyDrive/cosmic", "cosmic_OLD_do_not_use", "_OLD_100MB")  # never read or written


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


def _norm(path):
    return os.path.splitdrive(os.path.abspath(str(path)))[1].replace("\\", "/").rstrip("/") + "/"


def is_retired(path):
    """True if `path` is, or is inside, one of the retired folders."""
    p = _norm(path)
    old, *names = RETIRED
    return (old + "/") in p or any(n in p.split("/") for n in names)


def is_inside(path, folder):
    return _norm(path).startswith(_norm(folder))


@dataclass
class Config:
    settings: dict
    dataset: str

    @property
    def ds(self):
        return self.settings["datasets"][self.dataset]

    def _abs(self, path):
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    def _under(self, base, key):
        """A path from datasets.<name>.<key> if given (relative to the repo), else base/paths.<key>."""
        own = self.ds.get(key)
        return self._abs(own) if own else os.path.join(base, self.settings["paths"][key])

    def _env(self, name):
        return os.environ.get(name) if self.dataset == "real" else None

    @property
    def root(self):
        """PROJECT_ROOT: $COSMIC_PROJECT_ROOT (real data only), else datasets.<name>.project_root."""
        return self._abs(self._env("COSMIC_PROJECT_ROOT") or self.ds["project_root"])

    @property
    def v3_root(self):
        """V3_ROOT: $V3_ROOT (real data only), else PROJECT_ROOT/<paths.v3_subdir>. Every write goes here."""
        return self._abs(self._env("V3_ROOT") or os.path.join(self.root, self.settings["paths"]["v3_subdir"]))

    # ---- read-only inputs under PROJECT_ROOT
    @property
    def data_dir(self):
        """PROJECT_ROOT/data: the only folder read for AAVSO CSVs (not recursive)."""
        return self._under(self.root, "data_subdir")

    raw_dir = data_dir  # older name, kept for callers

    @property
    def labels_file(self):
        return self._under(self.root, "labels_file")

    @property
    def cache_dir(self):
        """Tarball cache for setup/install_stack.sh."""
        return self._under(self.root, "cache_subdir")

    # ---- everything written, under V3_ROOT
    def _v3_dir(self, key, create=True):
        path = self._under(self.v3_root, key)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    @property
    def work_dir(self):
        return self._v3_dir("work_subdir")

    @property
    def output_dir(self):
        path = self._under(self.v3_root, "output_subdir")
        if self.dataset != "real" and "review2" in path.replace("\\", "/").split("/"):
            raise ValueError("Synthetic outputs must never be written to review2/")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def figures_dir(self):
        path = os.path.join(self.output_dir, "figures")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def ztf_raw_dir(self):
        return self._v3_dir("ztf_raw_subdir")

    @property
    def clean_dir(self):
        return self._v3_dir("clean_subdir")

    @property
    def dashboard_dir(self):
        return self._v3_dir("dashboard_subdir")

    def paths(self):
        """Every resolved path, without creating anything."""
        return {"PROJECT_ROOT": self.root, "data (AAVSO CSVs, read only)": self.data_dir,
                "tarball cache (read only)": self.cache_dir, "VSX labels file": self.labels_file,
                "V3_ROOT (all new writes)": self.v3_root,
                "outputs": self._under(self.v3_root, "output_subdir"),
                "work": self._under(self.v3_root, "work_subdir"),
                "ZTF raw": self._under(self.v3_root, "ztf_raw_subdir"),
                "clean": self._under(self.v3_root, "clean_subdir"),
                "dashboard data": self._under(self.v3_root, "dashboard_subdir")}

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


WRITE_KEYS = ("outputs", "work", "ZTF raw", "clean", "dashboard data")


def check_paths(cfg, need_data=True):
    """Print the resolved paths; raise a clear error if any points into a retired folder, if a write
    folder is outside V3_ROOT, or if the data folder is missing. Returns the paths."""
    paths = cfg.paths()
    width = max(len(k) for k in paths)
    for k, v in paths.items():
        print(f"  {k:{width}s}  {v}")
    bad = {k: v for k, v in paths.items() if is_retired(v)}
    if bad:
        raise ValueError("These paths point at a retired folder (MyDrive/cosmic, cosmic_OLD_do_not_use or _OLD_100MB), "
                         f"which must never be used: {bad}. Set PROJECT_ROOT to /content/drive/MyDrive/cosmic_roja.")
    if cfg.dataset == "real":
        outside = {k: paths[k] for k in WRITE_KEYS if not is_inside(paths[k], paths["V3_ROOT (all new writes)"])}
        if outside:
            raise ValueError(f"These write folders are outside V3_ROOT, which is not allowed: {outside}")
    if need_data and not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"The data folder does not exist: {cfg.data_dir}\n"
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
