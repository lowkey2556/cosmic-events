"""The shared colour and naming config (config/style.yaml) for charts, tables and dashboard tables."""
import functools
import os

import yaml

from .config import REPO_ROOT


@functools.lru_cache(maxsize=1)
def load():
    with open(os.path.join(REPO_ROOT, "config", "style.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_name(key):
    return load()["models"].get(key, {}).get("name", key)


def model_color(key):
    return load()["models"].get(key, {}).get("color", "#7a7a7a")


def variant_name(key):
    return load()["variants"].get(key, key)
