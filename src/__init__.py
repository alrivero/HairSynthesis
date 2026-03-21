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


_ensure_hairstep_on_path()

