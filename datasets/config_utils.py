"""Helpers for accessing dataset-specific configuration blocks."""
from __future__ import annotations

from typing import Any, Optional


def get_dataset_section(config: Any, section_name: Optional[str] = None) -> Any:
    """Return the dataset config for `section_name`, falling back to the root.

    Works with OmegaConf DictConfig objects as well as plain dict / attribute
    containers. If the requested section does not exist, the top-level dataset
    config (or the original object) is returned to preserve backwards
    compatibility with the previous flat layout.
    """
    dataset_cfg = _get_dataset_root(config)
    if not section_name:
        return dataset_cfg

    for accessor in (_get_from_attr, _get_from_item, _get_from_get):
        value = accessor(dataset_cfg, section_name)
        if value is not None:
            return value

    return dataset_cfg


def _get_dataset_root(config: Any) -> Any:
    if hasattr(config, "dataset"):
        try:
            return getattr(config, "dataset")
        except Exception:
            pass
    return config


def _get_from_attr(cfg: Any, key: str) -> Any:
    try:
        return getattr(cfg, key)
    except Exception:
        return None


def _get_from_item(cfg: Any, key: str) -> Any:
    try:
        return cfg[key]
    except Exception:
        return None


def _get_from_get(cfg: Any, key: str) -> Any:
    get_fn = getattr(cfg, "get", None)
    if callable(get_fn):
        try:
            return get_fn(key, None)
        except Exception:
            return None
    return None
