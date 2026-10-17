"""Thin wrappers around the upstream planners, including the swap primitive.

Three jobs:

1. **Load the pretrained encoder without importing PyTorch Lightning.**
   Upstream's loader, `neural_astar.utils.training.load_from_ptl_checkpoint`,
   only calls `torch.load` and a regex, but it lives in a module whose first
   lines do ``import pytorch_lightning as pl``. Since the project forbids editing
   the submodule, `load_ptl_state_dict` reimplements those few lines here. That
   keeps every inference path (E0-E3, the flip sweeps) free of Lightning, which
   matters because Lightning 1.8 is the most fragile pin in the environment.

2. **Expose the cost map** ("guidance map"), the bottleneck this whole project
   studies, as a plain tensor (`encode_cost_map`).

3. **Call the differentiable A* module directly** with an arbitrary
   ``(cost map, obstacle map)`` pair (`run_astar_module`). This is the
   structural premise of the swap design (PLAN.md step 1 / E3): in
   `neural_astar/planner/differentiable_astar.py` the module takes
   ``cost_maps`` and ``obstacles_maps`` as separate arguments (lines 150-156)
   and the obstacle map is used only as a multiplicative mask on candidate
   neighbours (line 229). Nothing else in the module reads it.
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.planner.differentiable_astar import AstarOutput

from . import config

#: Canonical ordering of the 2x2 swap conditions (working rules, "Swap conditions"):
#: ``C{costs}O{obstacles}`` where ``0`` = original and ``1`` = altered.
SWAP_CONDITIONS = ("C0O0", "C0O1", "C1O0", "C1O1")


def load_ptl_state_dict(checkpoint_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Extract a planner `state_dict` from a PyTorch-Lightning checkpoint.

    Behaviourally identical to
    `neural_astar.utils.training.load_from_ptl_checkpoint`: if a directory is
    given, the lexicographically last ``*.ckpt`` below it is used, and keys are
    stripped of the ``planner.`` prefix that `PlannerModule` adds.

    Args:
        checkpoint_path: Either a ``.ckpt`` file or a directory containing one
            (searched recursively).

    Returns:
        dict[str, torch.Tensor]: Weights ready for
        `torch.nn.Module.load_state_dict`.

    Raises:
        FileNotFoundError: If no checkpoint is found under `checkpoint_path`.

    Note:
        Loaded with ``map_location="cpu"``; the caller moves the model to the
        device from `src.config.get_device`.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        matches = sorted(glob(f"{path}/**/*.ckpt", recursive=True))
        if not matches:
            raise FileNotFoundError(f"no *.ckpt found under {path}")
        ckpt_file = Path(matches[-1])
    else:
        ckpt_file = path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_file} does not exist")

    state_dict = torch.load(ckpt_file, map_location="cpu")["state_dict"]
    extracted = {
        re.split("planner.", key)[-1]: value
        for key, value in state_dict.items()
        if "planner" in key
    }
    if not extracted:
        raise FileNotFoundError(
            f"{ckpt_file} contains no keys prefixed with 'planner.'; "
            "is this a Neural A* checkpoint?"
        )
    return extracted


def load_pretrained_neural_astar(
    checkpoint_path: Optional[Union[str, Path]] = None,
    encoder_arch: str = "CNN",
    encoder_depth: int = 4,
    encoder_input: str = "m+",
    g_ratio: float = 0.5,
    device=None,
) -> NeuralAstar:
    """Instantiate Neural A* and load upstream's pretrained maze weights.

    The defaults are the configuration the shipped checkpoint was trained with
    (`neural-astar/scripts/config/train.yaml` and the `NeuralAstar` defaults):
    a 4-block plain CNN encoder on the two-channel input
    ``[map_design, start_map + goal_map]``, with ``g_ratio = 0.5``.

    Args:
        checkpoint_path: Checkpoint file or directory; defaults to
            `src.config.PRETRAINED_CKPT_DIR`.
        encoder_arch: Encoder class name in `neural_astar.planner.encoder`.
        encoder_depth: Number of conv blocks in the encoder.
        encoder_input: ``"m"`` for the map alone, ``"m+"`` to also feed
            ``start_map + goal_map``.
        g_ratio: Weight of ``g`` in ``f = g_ratio * g + (1 - g_ratio) * h``.
        device: Torch device; defaults to `src.config.get_device()`.

    Returns:
        NeuralAstar: The planner in ``eval()`` mode (which also forces the
        module's ``Tmax`` back to 1.0, i.e. no training-time search budget).

    Raises:
        RuntimeError: If the checkpoint's weights do not match the requested
            architecture (propagated from `load_state_dict`).
    """
    device = device or config.get_device()
    planner = NeuralAstar(
        g_ratio=g_ratio,
        encoder_input=encoder_input,
        encoder_arch=encoder_arch,
        encoder_depth=encoder_depth,
    ).to(device)
    planner.load_state_dict(
        load_ptl_state_dict(checkpoint_path or config.PRETRAINED_CKPT_DIR)
    )
    planner.eval()
    return planner


def make_vanilla_astar(g_ratio: float = 0.5, device=None) -> VanillaAstar:
    """Instantiate the upstream classical A* control.

    `VanillaAstar` runs the *same* differentiable A* module with the obstacle
    map used as the cost map (`neural_astar/planner/astar.py:93-94`), i.e. unit
    cost on every free cell. It is the "classical A* with the same heuristic"
    baseline that the Exp metric is defined against in the project's definitions.

    Args:
        g_ratio: Weight of ``g`` in ``f``; must match the Neural A* planner it
            is compared against.
        device: Torch device; defaults to `src.config.get_device()`.

    Returns:
        VanillaAstar: The planner in ``eval()`` mode.
    """
    device = device or config.get_device()
    planner = VanillaAstar(g_ratio=g_ratio).to(device)
    planner.eval()
    return planner


@torch.no_grad()
def encode_cost_map(
    planner: NeuralAstar,
    map_designs: torch.Tensor,
    start_maps: torch.Tensor,
    goal_maps: torch.Tensor,
) -> torch.Tensor:
    """Run only the encoder and return the predicted cost map.

    This is the object of study: a per-cell cost in ``(0, 1)`` (the encoder
    ends in a sigmoid, `neural_astar/planner/encoder.py:33-35`).

    Args:
        planner: A `NeuralAstar` instance.
        map_designs: ``(B, 1, H, W)`` binary maps, ``1`` = free.
        start_maps: ``(B, 1, H, W)`` one-hot start maps.
        goal_maps: ``(B, 1, H, W)`` one-hot goal maps.

    Returns:
        torch.Tensor: ``(B, 1, H, W)`` cost maps, detached.

    Note:
        Gradients are disabled here. The gradient-guided attack of PLAN.md
        step 9 needs them, and must call `planner.encode` directly instead.
    """
    return planner.encode(map_designs, start_maps, goal_maps)


@torch.no_grad()
def run_astar_module(
    planner: Union[NeuralAstar, VanillaAstar],
    cost_maps: torch.Tensor,
    start_maps: torch.Tensor,
    goal_maps: torch.Tensor,
    obstacles_maps: torch.Tensor,
    store_intermediate_results: bool = False,
) -> AstarOutput:
    """Search with an arbitrary (cost map, obstacle map) pair — the swap primitive.

    The two arguments are independent inputs of the differentiable A* module,
    so they may come from *different* maps. That is what makes the 2x2 swap
    design of E3 possible without touching the encoder:

    ======  ===================  =====================
    label   cost map from        obstacle mask from
    ======  ===================  =====================
    C0O0    original map         original map
    C0O1    original map         altered map
    C1O0    altered map          original map
    C1O1    altered map          altered map
    ======  ===================  =====================

    Args:
        planner: Any planner exposing ``.astar`` (both `NeuralAstar` and
            `VanillaAstar` do); only its search module is used, never its
            encoder, so the cost map must be supplied by the caller.
        cost_maps: ``(B, 1, H, W)`` per-cell costs. For a classical-A*
            control, pass the obstacle map itself (unit cost on free cells).
        start_maps: ``(B, 1, H, W)`` one-hot start maps.
        goal_maps: ``(B, 1, H, W)`` one-hot goal maps.
        obstacles_maps: ``(B, 1, H, W)`` binary mask, ``1`` = traversable.
        store_intermediate_results: Forwarded to the module; keeps the
            per-step search history (used for the animations only).

    Returns:
        AstarOutput: ``histories`` (``(B, 1, H, W)``, 1 where a node was
        expanded) and ``paths`` (``(B, 1, H, W)``, 1 on the returned path).

    Warning:
        An off-diagonal pair is off-distribution by construction: the encoder
        never produced that cost map for that obstacle mask. PLAN.md section 5
        lists this as a known validity risk, not a bug.
    """
    return planner.astar(
        cost_maps,
        start_maps,
        goal_maps,
        obstacles_maps,
        store_intermediate_results,
    )
