"""client.py
"""
from collections import OrderedDict
import logging
from typing import Any, Callable, ClassVar, Dict, Union, List
from copy import deepcopy
from torch._C import device
from threading import Lock

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .helpers import bundle_parameter


class ClientSimBackend(object):
    def __init__(self, id: int, net: nn.Module, device: Union[str, torch.device]) -> None:
        """Backend of a client. Since we only have limited GPUs.
        Multiple client can share one backend.
        

        Args:
            id (int): backend id
            net (nn.Module): backend net
            device (Union[str, torch.device]): backend device
        """
        super().__init__()
        self.id: int = id
        self.net: nn.Module = deepcopy(net.cpu())
        self.device: torch.device = device if isinstance(device, torch.device) else torch.device(device)
        self.net.to(self.device)
        self.lock = Lock()

    def __repr__(self) -> str:
        return f'<class: ClientSimBackend, id: {self.id}, device:{self.device}>'

    @property
    def parameters(self):
        return bundle_parameter(self.net)

    def __call__(self, dataset: Dataset, n_epochs: int, batch_sz: int, lr: float, criterion: Callable,
                 optim: torch.optim.Optimizer) -> Any:
        if self.net is None:
            logging.warn(f'clientbackend {self.id} has not initialized net')
            return
        # Acquire lock to make sure backend is not be used by more than one clients
        self.lock.acquire()

        # Begin of standard trainning process
        self.net.to(self.device)
        self.net.train()


        self.length = len(dataset)
        trainloader = DataLoader(dataset=dataset, batch_size=batch_sz, shuffle=True, num_workers=0)

        optimizer: torch.optim.Optimizer = optim(self.net.parameters(), lr=lr)
        with tqdm.tqdm(len(trainloader) * n_epochs) as pbar:
            for epoch_idx in range(n_epochs):
                for stimulis, labels in trainloader:
                    pred: torch.Tensor = self.net(stimulis.to(self.device))
                    loss: torch.Tensor = criterion(pred, labels.to(self.device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_description(f'id: {self.id}, epoch: {epoch_idx}, loss: {str(loss.detach().cpu().numpy())[:6]}')
                    pbar.update()

        # Delete trainloader and release backend
        del trainloader
        self.lock.release()

    def __len__(self):
        return self.length

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'new_parameters':
            self.lock.acquire()
            if isinstance(value, nn.Module):
                self.net.load_state_dict(deepcopy(value.state_dict()))
                self.net.to(self.device)
                self.lock.release()
                return
            elif isinstance(value, dict) or isinstance(value, OrderedDict):
                self.net.load_state_dict(deepcopy(value))
                self.net.to(self.device)
                self.lock.release()
                return
            self.lock.release()
            
        return super().__setattr__(name, value)


class ClientSim(object):
    def __init__(self, id: int, backend: ClientSimBackend, n_epochs: int, batch_sz: int, lr: float, criterion: nn.Module,
                 optim: torch.optim.Optimizer) -> None:
        """Client class. Clients are associated with backends

        Args:
            id (int): client id
            backend (ClientSimBackend): client backend
            n_epochs (int): number of epochs
            batch_sz (int): client batch size
            lr (float): client learning rate
            criterion (nn.Module): client loss function
            optim (torch.optim.Optimizer): client optimizer
        """
        super().__init__()
        self.id: int = id
        self.backend: ClientSimBackend = backend
        self.n_epochs: int = n_epochs
        self.batch_sz: int = batch_sz
        self.lr: float = lr
        self.criterion: Callable = criterion
        self.optim: torch.optim.Optimizer = optim
        self.length: int = 0

    def __repr__(self) -> str:
        return f'<class: ClientSim, id: {self.id}, device:{self.backend.device}>'

    @property
    def device(self) -> torch.device:
        return self.backend.device
    
    @property
    def parameters(self) -> Dict[str, torch.Tensor]:
        return self.backend.parameters

    def __call__(self, dataset: Dataset) -> Any:
        self.length = len(dataset)

        return self.backend(dataset, self.n_epochs, self.batch_sz, self.lr, self.criterion, self.optim)

    def __len__(self):
        return self.length

    def __setattr__(self, name: str, value: Any) -> None:
        # Setting client parameters is equivalent to setting backend parameters
        if name == 'new_parameters':
            self.backend.new_parameters = value
        return super().__setattr__(name, value)