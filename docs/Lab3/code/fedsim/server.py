"""server.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import tqdm
from copy import deepcopy

from typing import Any, ClassVar, Dict, Hashable, List, Tuple, Union, Type
from .helpers import bundle_parameter


class ServerSim(object):
    def __init__(self, net: nn.Module, device: Union[str, torch.device]) -> None:
        super().__init__()
        self.net: nn.Module = net
        self.device: torch.device = device if isinstance(device, torch.device) else torch.device(device)
        self.net.to(device)
        self.net.eval()
        self.cached_params: List[List[float, Dict[str, torch.Tensor]]] = []
        self.empty_params: OrderedDict = OrderedDict()

        for name, param in list(self.net._named_members(lambda module: module._parameters.items())):
            self.empty_params[name] = torch.zeros_like(param)

    @property
    def parameters(self):
        return bundle_parameter(self.net)

    @torch.no_grad()
    def __call__(self) -> None:
        if len(self.cached_params) <= 0:
            return None

        tot_samples: int = sum([self.cached_params[i][0] for i in range(len(self.cached_params))])
        print(f'[ Info ] Total number of samples: {tot_samples}')
        for idx in range(len(self.cached_params)):
            self.cached_params[idx][0] /= tot_samples

        # Prepare an OrderedDict for result
        new_params = deepcopy(self.empty_params)

        # Gather all model parameters
        for gain, cached_param in self.cached_params:
            for name in new_params.keys():
                new_params[name] += cached_param[name].to(self.device) * gain

        self.net.load_state_dict(new_params)
        print('[ Debug ] New params loaded')

        # Clean cache
        self.cached_params = []

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.cached_params.append([key, value])
        return None
