"""mapping:
A backend module used to map memristive conductances to PyTorch synaptic layers, and vice versa.
"""
import torch


def weights_to_conductances(
    weights: torch.Tensor,
    G_off: float,
    G_on: float,
    mapping_style: str = "lowest",
) -> torch.Tensor:
    """Map PyTorch weights onto conductances.

    Args:
        weights: The original layer of weights as a Tensor.
        G_off: A float representing the conductance of the devices in their OFF state.
        G_on: A float representing the conductance of the devices in their ON state.
        mapping_style: A string, either `lowest` or `average`.
            Represents the mapping style to be utilised, where
            `lowest` instructs the function to map according to the lowest possible conductance, and
            `average` instructs the function to map the weights around an average conductance.

    Returns:
        conductance_layer: A PyTorch Tensor representing the mapped conductances, shape `m x 2n`.
        maximum_weight: The max weight in the original weights layer.
    """

    maximum_weight = torch.max(torch.abs(weights))
    scaling_fact_weight = (G_on - G_off) / maximum_weight
    effective_cond = scaling_fact_weight * weights

    if mapping_style == "lowest":

        # Map according to lowest possible conductance
        cond_pos = torch.maximum(effective_cond, 0.0) + G_off
        cond_neg = -torch.maximum(effective_cond, 0.0) + G_off

        # PRETTY SURE THE ABOVE SHOULD BE TENSORS OF ZEROS

    elif mapping_style == "average":

        # Map around the average conductance
        avg_cond = (G_on + G_off) / 2
        cond_pos, cond_neg = avg_cond + 0.5 * effective_cond, avg_cond - 0.5 * effective_cond

    else:

        raise ValueError("mapping_style parameter not valid.")

    conductance_layer = torch.reshape(
        torch.cat((cond_pos[..., None], cond_neg[..., None]), dim=-1),
        [cond_pos.size(dim=0), -1],
    )

    return conductance_layer, maximum_weight
