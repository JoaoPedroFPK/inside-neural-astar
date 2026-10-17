"""Environment reproducibility checks.

These tests fail loudly if a colleague's environment differs from the one the
results were produced in. The pins they assert are the ones in
`requirements.txt` and in the upstream submodule's `pyproject.toml`; see
`docs/setup.md` for how the environment is built and why each pin exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_python_is_310() -> None:
    """Python must be 3.10.x: torch 1.12.1 publishes no wheels for 3.11+."""
    assert sys.version_info[:2] == (3, 10), (
        f"expected Python 3.10.x (see .python-version), got {sys.version.split()[0]}"
    )


def test_running_inside_project_venv() -> None:
    """The interpreter must be the project's `.venv`, not a global Python."""
    repo_root = Path(__file__).resolve().parents[1]
    assert Path(sys.prefix).resolve() == (repo_root / ".venv").resolve(), (
        f"expected the project venv at {repo_root / '.venv'}, got sys.prefix={sys.prefix}. "
        "Run `source .venv/bin/activate` (or `make test`)."
    )


@pytest.mark.parametrize(
    "module_name, attribute, expected",
    [
        ("torch", "__version__", "1.12.1"),
        ("torchvision", "__version__", "0.13.1"),
        ("numpy", "__version__", "1.26.4"),
        ("scipy", "__version__", "1.14.1"),
        ("sklearn", "__version__", "1.5.2"),
        ("pandas", "__version__", "2.2.3"),
        ("pytorch_lightning", "__version__", "1.8.5.post0"),
        ("segmentation_models_pytorch", "__version__", "0.3.1"),
    ],
)
def test_pinned_versions(module_name: str, attribute: str, expected: str) -> None:
    """Every pinned dependency must be at its pinned version."""
    module = pytest.importorskip(module_name)
    found = getattr(module, attribute)
    assert found.split("+")[0] == expected, (
        f"{module_name} is {found}, expected {expected}; reinstall with "
        "`bash scripts/setup_env.sh --recreate`"
    )


def test_numpy_abi_compatible_with_torch() -> None:
    """numpy must stay on 1.x: 2.x is ABI-incompatible with torch 1.12.1."""
    import numpy as np
    import torch

    assert int(np.__version__.split(".")[0]) == 1
    # Round-trip through the bridge that would segfault/raise on a mismatch.
    assert torch.from_numpy(np.zeros((2, 2), dtype=np.float32)).sum().item() == 0.0


def test_pkg_resources_available() -> None:
    """`pkg_resources` must exist: pytorch_lightning 1.8 imports it.

    setuptools >= 81 removed `pkg_resources`, which breaks
    `import pytorch_lightning` with a `ModuleNotFoundError`. The environment
    therefore pins setuptools 69.5.1.
    """
    import pkg_resources  # noqa: F401
    import setuptools

    major = int(setuptools.__version__.split(".")[0])
    assert major < 81, (
        f"setuptools {setuptools.__version__} removed pkg_resources, which "
        "pytorch_lightning 1.8.5 needs; pin setuptools==69.5.1"
    )


def test_device_defaults_to_cpu() -> None:
    """The single device source must default to CPU."""
    from src.config import get_device

    assert get_device().type == "cpu"


def test_submodule_present_and_unmodified() -> None:
    """The upstream submodule must be checked out and clean.

    The project forbids editing files inside `neural-astar/`; a dirty submodule
    means a result cannot be reproduced from the pinned commit.
    """
    import subprocess

    from src.config import REPO_ROOT, SUBMODULE_DIR

    assert (SUBMODULE_DIR / "src" / "neural_astar").is_dir(), (
        "submodule missing; run `git submodule update --init --recursive`"
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=SUBMODULE_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status == "", f"submodule has local modifications:\n{status}"
    # The nested data submodule is what `data/mazes_032_moore_c8.npz` points at.
    assert (REPO_ROOT / "data" / "mazes_032_moore_c8.npz").exists()


def test_pretrained_checkpoint_present() -> None:
    """Upstream's pretrained maze checkpoint must be available for E0."""
    from src.config import PRETRAINED_CKPT_DIR

    ckpts = sorted(PRETRAINED_CKPT_DIR.glob("**/*.ckpt"))
    assert ckpts, f"no checkpoint under {PRETRAINED_CKPT_DIR}"


def test_project_imports() -> None:
    """All first-party modules import cleanly."""
    import importlib

    for name in ("src.config", "src.grid", "src.data", "src.planner", "src.metrics"):
        importlib.import_module(name)
