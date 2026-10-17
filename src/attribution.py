"""Attribution: which input cells drive one cost-map value — E1 (PLAN.md step 8).

The surrogate of step 5 asks what the cost map *correlates with*. Attribution
asks a narrower, mechanistic question: for one chosen cell of the cost map,
which parts of the **input** did the encoder actually use?

Three methods, in increasing order of what they control for:

* `saliency` — the plain gradient of the chosen output with respect to the
  input. Cheap, and notoriously noisy: it measures sensitivity at one point,
  which can be large where the function is locally jagged but globally flat.
* `integrated_gradients` — the gradient averaged along a straight path from a
  *baseline* input to the real one, times the difference. It satisfies
  *completeness*: the attributions sum exactly to the change in output between
  baseline and input, so the numbers have a unit and a budget rather than being
  merely ordinal.
* `grad_cam` — gradient-weighted activations of the last convolutional layer.
  Coarser, but it answers "which features, where" rather than "which pixels".

And one control, which matters more than any of them:

* `randomization_check` — re-run the attribution with the encoder's weights
  randomised. An attribution that barely changes is not describing the trained
  model; it is describing the architecture or the input. This is the sanity
  check of Adebayo et al. (2018), "Sanity Checks for Saliency Maps", and it is
  the reason this module exists in a project about mechanism rather than
  pictures.

**On the baseline.** Integrated gradients needs an input representing
"absence", and there is no neutral choice here. A zero map means *every cell is
an obstacle* (the maps encode free as 1); an all-ones map means every cell is
free. The two give different attributions for the same output, which is a
property of the method, not a bug. Both are provided; the default is
``"free"``, an obstacle-free map with the same start and goal, because it
isolates what the *obstacles* contribute.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import MazeInstance
from .grid import RC

#: Channel names of the encoder's input, in order. The encoder is built with
#: ``encoder_input="m+"``, so it receives the obstacle map and the sum of the
#: start and goal one-hot maps (`planner/astar.py:171-178`).
INPUT_CHANNELS: Tuple[str, str] = ("map_design", "start_plus_goal")


@dataclass
class Attribution:
    """Attribution of one cost-map value to the encoder's input.

    Attributes:
        target_rc: The cost-map cell explained.
        values: ``(2, H, W)`` attribution per input channel.
        method: Name of the method used.
        baseline: Name of the baseline, where the method has one.
        output_value: The cost-map value at `target_rc` for the real input.
        baseline_output: The same for the baseline input, where applicable.
        completeness_error: ``|sum(values) - (output - baseline_output)|``, the
            degree to which the completeness property is violated by the finite
            number of integration steps. Small is good; it is the honest
            accuracy measure for integrated gradients.
    """

    target_rc: RC
    values: np.ndarray
    method: str
    baseline: Optional[str] = None
    output_value: float = float("nan")
    baseline_output: float = float("nan")
    completeness_error: float = float("nan")

    def by_channel(self) -> Dict[str, np.ndarray]:
        """Return the attribution split by named input channel.

        Returns:
            dict[str, np.ndarray]: ``(H, W)`` array per channel name.
        """
        return {name: self.values[i] for i, name in enumerate(INPUT_CHANNELS)}

    def total_by_channel(self) -> Dict[str, float]:
        """Summed attribution per channel — which input mattered at all.

        Returns:
            dict[str, float]: Signed sum per channel.
        """
        return {name: float(array.sum()) for name, array in self.by_channel().items()}


def encoder_input(instance: MazeInstance, device=None) -> torch.Tensor:
    """Build the two-channel tensor the encoder actually receives.

    Args:
        instance: The problem instance.
        device: Torch device; defaults to the project device.

    Returns:
        torch.Tensor: ``(1, 2, H, W)`` input, channel 0 the obstacle map and
        channel 1 the sum of the start and goal one-hot maps.
    """
    maps, starts, goals = instance.to_tensors(device)
    return torch.cat([maps, starts + goals], dim=1)


def _baseline_input(inputs: torch.Tensor, baseline: str) -> torch.Tensor:
    """Build the baseline input for a path-integration method.

    Args:
        inputs: The real ``(1, 2, H, W)`` input.
        baseline: ``"free"`` (no obstacles, same start and goal), ``"zeros"``
            (everything an obstacle, no start or goal) or ``"blocked"`` (every
            cell an obstacle but start and goal kept).

    Returns:
        torch.Tensor: Baseline input of the same shape.

    Raises:
        ValueError: If `baseline` is unknown.
    """
    out = torch.zeros_like(inputs)
    if baseline == "zeros":
        return out
    if baseline == "free":
        out[:, 0] = 1.0
        out[:, 1] = inputs[:, 1]
        return out
    if baseline == "blocked":
        out[:, 1] = inputs[:, 1]
        return out
    raise ValueError(f"unknown baseline {baseline!r}; expected free, zeros or blocked")


def _target_output(planner, inputs: torch.Tensor, target_rc: RC) -> torch.Tensor:
    """Run the encoder and return the scalar cost at one cell.

    Args:
        planner: A `NeuralAstar`.
        inputs: ``(1, 2, H, W)`` encoder input.
        target_rc: Cell of the cost map to read.

    Returns:
        torch.Tensor: Scalar tensor, differentiable with respect to `inputs`.
    """
    cost = planner.encoder(inputs)
    return cost[0, 0, target_rc[0], target_rc[1]]


def saliency(planner, instance: MazeInstance, target_rc: RC) -> Attribution:
    """Plain input gradient of one cost-map value.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The problem instance.
        target_rc: Which cost-map cell to explain.

    Returns:
        Attribution: With `values` the signed gradient per input cell.
    """
    inputs = encoder_input(instance).requires_grad_(True)
    output = _target_output(planner, inputs, target_rc)
    (gradient,) = torch.autograd.grad(output, inputs)
    return Attribution(
        target_rc=target_rc,
        values=gradient[0].detach().numpy(),
        method="saliency",
        output_value=float(output.detach()),
    )


def integrated_gradients(
    planner,
    instance: MazeInstance,
    target_rc: RC,
    baseline: str = "free",
    steps: int = 64,
) -> Attribution:
    """Integrated gradients for one cost-map value.

    Approximates ``(x - x') * ∫ ∂F(x' + α(x - x'))/∂x dα`` with a Riemann sum
    at the midpoints of `steps` intervals.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The problem instance.
        target_rc: Which cost-map cell to explain.
        baseline: Baseline name, see `_baseline_input`.
        steps: Number of integration steps. The reported
            `completeness_error` says whether it was enough.

    Returns:
        Attribution: With `values` the per-cell contributions, which sum to the
        difference between the real and baseline outputs up to
        `completeness_error`.

    Note:
        Attribution is to the *encoder's* input only. The search that consumes
        the cost map is not differentiated here; its gradients are a separate
        question (and upstream detaches the accumulated cost, so the heuristic
        term is the only differentiable route through the search).
    """
    inputs = encoder_input(instance)
    base = _baseline_input(inputs, baseline)
    difference = inputs - base

    total = torch.zeros_like(inputs)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (base + alpha * difference).requires_grad_(True)
        output = _target_output(planner, point, target_rc)
        (gradient,) = torch.autograd.grad(output, point)
        total = total + gradient
    values = (difference * total / steps)[0].detach().numpy()

    with torch.no_grad():
        real_output = float(_target_output(planner, inputs, target_rc))
        base_output = float(_target_output(planner, base, target_rc))
    return Attribution(
        target_rc=target_rc,
        values=values,
        method="integrated_gradients",
        baseline=baseline,
        output_value=real_output,
        baseline_output=base_output,
        completeness_error=float(abs(values.sum() - (real_output - base_output))),
    )


def grad_cam(planner, instance: MazeInstance, target_rc: RC, layer: int = -2) -> Attribution:
    """Gradient-weighted activation map of one convolutional layer.

    Each channel of the chosen layer's activation is weighted by the mean
    gradient of the target output with respect to that channel, the weighted
    sum is taken, and negative values are clipped — the standard Grad-CAM
    recipe. Because this encoder never downsamples, the result is already at
    full map resolution and needs no interpolation.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The problem instance.
        target_rc: Which cost-map cell to explain.
        layer: Index into `planner.encoder.model` of the layer whose output to
            use. The default −2 is the last batch-norm before the sigmoid.

    Returns:
        Attribution: With `values` of shape ``(1, H, W)`` — a single map rather
        than one per input channel.
    """
    inputs = encoder_input(instance)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        """Capture the layer's output so its gradient can be taken later."""
        output.retain_grad()
        captured["activation"] = output

    handle = planner.encoder.model[layer].register_forward_hook(hook)
    try:
        output = _target_output(planner, inputs, target_rc)
        activation = captured["activation"]
        (gradient,) = torch.autograd.grad(output, activation)
    finally:
        handle.remove()

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    return Attribution(
        target_rc=target_rc,
        values=cam[0].detach().numpy(),
        method=f"grad_cam[{layer}]",
        output_value=float(output.detach()),
    )


def randomize_encoder(planner, seed: int = 0, cascading_from: Optional[int] = None):
    """Return a copy of the planner with (part of) the encoder re-initialised.

    Args:
        planner: The trained `NeuralAstar`.
        seed: Seed for the re-initialisation.
        cascading_from: If given, only layers with index ≥ this value are
            randomised (the "cascading" variant of the sanity check, which
            degrades the model progressively from the output backwards). When
            omitted, every layer is randomised.

    Returns:
        NeuralAstar: A deep copy with randomised encoder weights, in eval mode.
        The original is untouched.
    """
    randomized = copy.deepcopy(planner)
    generator = torch.Generator().manual_seed(seed)
    for index, module in enumerate(randomized.encoder.model):
        if cascading_from is not None and index < cascading_from:
            continue
        if hasattr(module, "weight") and module.weight is not None:
            with torch.no_grad():
                if module.weight.dim() > 1:
                    bound = 1.0 / np.sqrt(np.prod(module.weight.shape[1:]))
                    module.weight.uniform_(-bound, bound, generator=generator)
                else:
                    module.weight.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None:
            with torch.no_grad():
                module.bias.zero_()
    randomized.eval()
    return randomized


def randomization_check(
    planner,
    instance: MazeInstance,
    target_rc: RC,
    method: str = "integrated_gradients",
    seed: int = 0,
    **kwargs,
) -> Dict[str, float]:
    """Compare an attribution before and after randomising the encoder.

    The logic of Adebayo et al. (2018): an attribution method that produces a
    similar map for a trained and an untrained network is not explaining what
    the network learned. A **low** similarity is the passing result.

    Args:
        planner: The trained `NeuralAstar`.
        instance: The problem instance.
        target_rc: Which cost-map cell to explain.
        method: ``"saliency"``, ``"integrated_gradients"`` or ``"grad_cam"``.
        seed: Seed for the randomisation.
        **kwargs: Forwarded to the attribution function.

    Returns:
        dict[str, float]: ``pearson`` and ``spearman`` correlation between the
        two attribution maps, the cosine similarity of their absolute values,
        and both outputs' values at the target cell.

    Raises:
        ValueError: If `method` is unknown.
    """
    from scipy import stats

    functions = {
        "saliency": saliency,
        "integrated_gradients": integrated_gradients,
        "grad_cam": grad_cam,
    }
    if method not in functions:
        raise ValueError(f"unknown method {method!r}")
    function = functions[method]

    trained = function(planner, instance, target_rc, **kwargs)
    randomized_planner = randomize_encoder(planner, seed=seed)
    randomized = function(randomized_planner, instance, target_rc, **kwargs)

    a = trained.values.ravel()
    b = randomized.values.ravel()
    denominator = np.linalg.norm(np.abs(a)) * np.linalg.norm(np.abs(b))
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan"),
        "spearman": float(stats.spearmanr(a, b).statistic),
        "abs_cosine": float(np.dot(np.abs(a), np.abs(b)) / denominator)
        if denominator > 0
        else float("nan"),
        "trained_output": trained.output_value,
        "randomized_output": randomized.output_value,
    }


def attribution_radius(attribution: Attribution, channel: int = 0) -> Dict[str, float]:
    """How far from the explained cell the attribution mass sits.

    The encoder's receptive field has radius 5, so an attribution that places
    mass beyond that is reporting an influence the architecture cannot support
    — a direct, cheap falsification test for the method.

    Args:
        attribution: An attribution to measure.
        channel: Which input channel to measure, when there are several.

    Returns:
        dict[str, float]: ``max_radius`` with non-zero attribution,
        ``share_within_5`` of the absolute mass, and ``share_on_target``.
    """
    values = np.abs(attribution.values[channel])
    total = values.sum()
    if total == 0:
        return {"max_radius": 0.0, "share_within_5": float("nan"), "share_on_target": 0.0}
    rows, columns = np.indices(values.shape)
    radius = np.maximum(
        np.abs(rows - attribution.target_rc[0]), np.abs(columns - attribution.target_rc[1])
    )
    non_zero = values > 0
    return {
        "max_radius": float(radius[non_zero].max()),
        "share_within_5": float(values[radius <= 5].sum() / total),
        "share_on_target": float(values[attribution.target_rc] / total),
    }
