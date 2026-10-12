"""Configuration loading.

What it does: reads config/settings.yaml, applies any `--set key.sub=value`
overrides from the command line, and resolves the dataset-specific paths
(real vs synthetic) into one small `Config` object that every step uses.

Why: every path and parameter lives in one YAML file, so they can be
seen and changed without touching code. Real and synthetic data are kept
apart by construction (different HDFS folder, Hive database and output folder),
so synthetic results can never leak into the Review 2 deliverables.
"""
import copy
import os
from dataclasses import dataclass

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


@dataclass
class Config:
    settings: dict
    dataset: str

    @property
    def ds(self):
        return self.settings["datasets"][self.dataset]

    def _abs(self, path):
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    @property
    def root(self):
        return self._abs(self.ds["root"])

    @property
    def raw_dir(self):
        """Folder searched recursively for AAVSO CSVs: $COSMIC_DATA_ROOT (real data only),
        else datasets.<name>.data_root, else <root>/raw."""
        env = os.environ.get("COSMIC_DATA_ROOT") if self.dataset == "real" else None
        path = env or self.ds.get("data_root")
        return self._abs(path) if path else os.path.join(self.root, self.settings["paths"]["raw_subdir"])

    @property
    def labels_file(self):
        return os.path.join(self.root, self.settings["paths"]["labels_file"])

    @property
    def output_dir(self):
        sub = self.ds["output_subdir"]
        if self.dataset != "real" and "review2" in sub.replace("\\", "/").split("/"):
            raise ValueError("Synthetic outputs must never be written to review2/")
        if self.dataset == "real":
            path = os.path.join(self.root, sub)
        else:
            path = os.path.join(REPO_ROOT, sub)
        os.makedirs(path, exist_ok=True)
        return path

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
        with open(self.repo_path(self.settings["paths"][key])) as fh:
            return yaml.safe_load(fh)


def load_config(dataset="real", overrides=None, path=None):
    path = path or os.path.join(REPO_ROOT, "config", "settings.yaml")
    with open(path) as fh:
        settings = yaml.safe_load(fh)
    settings = apply_overrides(settings, overrides)
    if dataset not in settings["datasets"]:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return Config(settings=settings, dataset=dataset)
