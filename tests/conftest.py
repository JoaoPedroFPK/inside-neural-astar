"""Shared fixtures.

Puts the repository root on `sys.path` so that `import src...` works without
installing this project as a package, and loads the heavy objects (the maze
split, the pretrained planner) once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def test_split():
    """The 100-instance test split, with distances sign-normalised."""
    from src.data import load_split

    return load_split("test")


@pytest.fixture(scope="session")
def pretrained_planner():
    """Neural A* with upstream's pretrained maze weights, in eval mode."""
    from src.planner import load_pretrained_neural_astar

    return load_pretrained_neural_astar()


@pytest.fixture(scope="session")
def vanilla_planner():
    """Classical A* (same module, unit costs), in eval mode."""
    from src.planner import make_vanilla_astar

    return make_vanilla_astar()
