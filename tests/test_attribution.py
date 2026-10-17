"""PLAN.md step 8: attribution of a cost-map value to the encoder's input."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.attribution import (
    INPUT_CHANNELS,
    Attribution,
    attribution_radius,
    encoder_input,
    grad_cam,
    integrated_gradients,
    randomization_check,
    randomize_encoder,
    saliency,
)

TARGET = (14, 17)


@pytest.fixture(scope="module")
def instance(test_split):
    """One test instance with a fixed start."""
    return test_split.instance(0, seed=0)


def test_encoder_input_has_the_two_channels_the_encoder_expects(instance) -> None:
    """Channel 0 is the obstacle map; channel 1 is start + goal."""
    inputs = encoder_input(instance)
    assert inputs.shape == (1, 2, *instance.map_design.shape)
    assert INPUT_CHANNELS == ("map_design", "start_plus_goal")
    assert np.array_equal(inputs[0, 0].numpy(), instance.map_design)
    assert inputs[0, 1].sum().item() == 2.0  # one start plus one goal
    assert inputs[0, 1][instance.start_rc].item() == 1.0
    assert inputs[0, 1][instance.goal_rc].item() == 1.0


def test_integrated_gradients_satisfies_completeness(pretrained_planner, instance) -> None:
    """Attributions sum to the output change, up to the integration error.

    Completeness is what gives integrated gradients a unit: the numbers are
    shares of an actual change in the cost value, not arbitrary scores. The
    tolerance here is 5 % of the change being explained.
    """
    result = integrated_gradients(pretrained_planner, instance, TARGET, steps=64)
    change = result.output_value - result.baseline_output
    assert abs(change) > 1e-3, "baseline and input must differ for the test to mean anything"
    assert result.completeness_error < 0.05 * abs(change)


def test_integrated_gradients_respects_the_receptive_field(
    pretrained_planner, instance
) -> None:
    """No attribution mass may fall outside radius 5 of the explained cell.

    The encoder is five 3x3 convolutions, so input cells beyond that radius
    cannot influence the output at all. An attribution that puts mass there
    would be reporting an influence the architecture cannot support.
    """
    result = integrated_gradients(pretrained_planner, instance, TARGET, steps=32)
    radius = attribution_radius(result)
    assert radius["max_radius"] <= 5.0
    assert radius["share_within_5"] == 1.0


def test_saliency_also_respects_the_receptive_field(pretrained_planner, instance) -> None:
    """The plain gradient is bounded by the same architectural fact."""
    radius = attribution_radius(saliency(pretrained_planner, instance, TARGET))
    assert radius["max_radius"] <= 5.0


def test_free_baseline_gives_the_start_goal_channel_no_credit(
    pretrained_planner, instance
) -> None:
    """With the `free` baseline, start and goal are unchanged, so they get zero.

    Not a defect: it is what the baseline means. Attributing to start and goal
    requires a baseline that removes them, which is why `blocked` and `zeros`
    exist.
    """
    free = integrated_gradients(pretrained_planner, instance, TARGET, baseline="free", steps=16)
    assert free.total_by_channel()["start_plus_goal"] == 0.0
    blocked = integrated_gradients(
        pretrained_planner, instance, TARGET, baseline="blocked", steps=16
    )
    assert blocked.total_by_channel()["map_design"] != 0.0


def test_unknown_baseline_is_rejected(pretrained_planner, instance) -> None:
    """A typo in the baseline name fails loudly."""
    with pytest.raises(ValueError, match="unknown baseline"):
        integrated_gradients(pretrained_planner, instance, TARGET, baseline="blank")


def test_grad_cam_is_a_single_non_negative_map(pretrained_planner, instance) -> None:
    """Grad-CAM returns one map at full resolution, with negatives clipped."""
    result = grad_cam(pretrained_planner, instance, TARGET)
    assert result.values.shape == (1, *instance.map_design.shape)
    assert (result.values >= 0).all()


def test_randomization_leaves_the_trained_planner_untouched(
    pretrained_planner, instance
) -> None:
    """The sanity check must not corrupt the model every other test shares."""
    before = pretrained_planner.encoder.model[0].weight.detach().clone()
    randomized = randomize_encoder(pretrained_planner, seed=0)
    after = pretrained_planner.encoder.model[0].weight.detach()
    assert torch.equal(before, after), "randomize_encoder mutated the original planner"
    assert not torch.equal(before, randomized.encoder.model[0].weight.detach())


def test_randomization_check_reports_low_similarity(pretrained_planner, instance) -> None:
    """Attribution on the trained model must differ from the randomised one.

    This is the Adebayo et al. sanity check: a method whose output survives
    weight randomisation is describing the architecture, not what was learned.
    A low correlation is the passing result.
    """
    scores = randomization_check(
        pretrained_planner, instance, TARGET, method="integrated_gradients", steps=16
    )
    assert abs(scores["pearson"]) < 0.5, f"attribution barely changed: {scores}"
    assert np.isfinite(scores["spearman"])


def test_attribution_radius_on_a_known_map() -> None:
    """The radius helper measures what it claims on a hand-made attribution."""
    values = np.zeros((2, 11, 11))
    values[0, 5, 5] = 1.0   # on the target
    values[0, 5, 8] = 1.0   # three cells away
    result = Attribution(target_rc=(5, 5), values=values, method="test")
    radius = attribution_radius(result)
    assert radius["max_radius"] == 3.0
    assert radius["share_on_target"] == 0.5
    assert radius["share_within_5"] == 1.0
