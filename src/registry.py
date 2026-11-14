"""Decorator-based component registries."""
from __future__ import annotations
from typing import Callable, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable] = {}

    def register(self, key: str):
        def deco(obj):
            if key in self._items:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._items[key] = obj
            return obj
        return deco

    def get(self, key: str) -> Callable:
        if key not in self._items:
            raise KeyError(f"{self.name}: unknown '{key}'. Available: {sorted(self._items)}")
        return self._items[key]

    def build(self, key: str, **kwargs):
        return self.get(key)(**kwargs)

    def keys(self):
        return sorted(self._items)

    def __contains__(self, key):
        return key in self._items


DYNAMICS = Registry("dynamics")
OBJECTIVES = Registry("objectives")
POLICIES = Registry("bank_policies")
REGULATORS = Registry("regulators")
