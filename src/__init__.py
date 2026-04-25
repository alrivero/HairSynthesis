"""Project-level package initialization for HairSynthesis."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_hairstep_on_path() -> None:
    """Adds external/HairStep to sys.path so `lib.*` imports resolve."""
    repo_root = Path(__file__).resolve().parent.parent
    hair_step = repo_root / "external" / "HairStep"
    if hair_step.exists():
        hair_step_str = str(hair_step)
        if hair_step_str not in sys.path:
            sys.path.insert(0, hair_step_str)


def _ensure_perm_on_path() -> None:
    """Adds external/perm/src to sys.path so PERM's top-level imports resolve."""
    repo_root = Path(__file__).resolve().parent.parent
    perm_src = repo_root / "external" / "perm" / "src"
    if perm_src.exists():
        perm_src_str = str(perm_src)
        if perm_src_str not in sys.path:
            sys.path.insert(0, perm_src_str)


_ensure_hairstep_on_path()
_ensure_perm_on_path()
