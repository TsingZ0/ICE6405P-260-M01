from typing import Dict, List, Any

import torch
import torch.nn as nn


def bundle_parameter(net: nn.Module) -> Dict[str, torch.Tensor]:
    """Bundle modle parameters to a dictionary, CUDA parameters will be copied to CPU

    Args:
        net (nn.Module): [description]

    Returns:
        Dict[str, torch.Tensor]: [description]
    """
    parameter_dict = {}
    module_parameters: List[str, nn.parameter.Parameter] = list(net._named_members(lambda module: module._parameters.items()))
    for name, param in module_parameters:
        parameter_dict[name] = param.clone().detach().cpu()
    return parameter_dict


def partition(ls: List[Any], size: int):
    """Returns a new list with elements of which is a list of certain size.

    >>> partition([1, 2, 3, 4, 5], 3)
        [[1, 2, 3], [4, 5]]
    """
    return [ls[i:i + size] for i in range(0, len(ls), size)]

