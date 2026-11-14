"""YAML config loading with `base:` inheritance and dot-notation overrides."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""
    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in self.items()}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base:
        raw = deep_merge(_load_raw((path.parent / base).resolve()), raw)
    return raw


def apply_overrides(d: dict, overrides) -> dict:
    """overrides: iterable of 'a.b.c=value' (value parsed as YAML)."""
    d = copy.deepcopy(d)
    for ov in overrides or []:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"Override must look like a.b=c, got '{ov}'")
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):                      # PyYAML reads '1e-3' as a string
            try:
                v = float(v)
            except ValueError:
                pass
        node[parts[-1]] = v
    return d


def load_config(path, overrides=None) -> Cfg:
    raw = _load_raw(Path(path).resolve())
    return Cfg(apply_overrides(raw, overrides))
