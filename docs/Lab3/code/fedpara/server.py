import pickle
from hashlib import md5
from typing import Dict, List
from collections import OrderedDict
import time
import os

import torch
import torch.nn as nn

from .conn import ConnABC
from .config import *
from .helpers import bundle_parameter, verify_state_dict

class ServerABC(object):
    def __init__(self, *args, **kwargs) -> None:
        """Server abstraction
        """
        super().__init__()

        # Server information maintained
        self.net: nn.Module = None
        self.params_size: int = None
        self.shared_params: ConnABC = None
        self.status: int = SIG_INIT

        # Client information maintained
        self.client_ids: List[int] = []
        self.client_params: Dict[int, ConnABC] = dict()
        self.client_info: Dict[int, ConnABC] = dict()
        self.client_signal: Dict[int, ConnABC] = dict()

    def get_client_params_path(self, id: int) -> str:
        return os.path.join(MMAP_PATH, CLIENT_PARAMS_FILEDESC.format(id))

    def get_client_signal_path(self, id: int) -> str:
        return os.path.join(MMAP_PATH, CLIENT_SIGNAL_FILEDESC.format(id))

    def get_client_info_path(self, id: int) -> str:
        return os.path.join(MMAP_PATH, CLIENT_INFO_FILEDESC.format(id))

    def get_client_params(self, id: int) -> OrderedDict:
        """Get model parameters from client[id]

        Args:
            id (int): Unique id of client

        Returns:
            OrderedDict: Model paramters
        """
        params = self.client_params[id].get()
        return params

    def get_client_signal(self, id: int) -> int:
        """Get signal from client[id]

        Args:
            id (int): Unique id of client

        Returns:
            int: Signal
        """
        signal = self.client_signal[id].get(decode=False)[0]
        return signal

    def get_client_info(self, id: int) -> int:
        """Get info from client[id]

        Args:
            id (int): Unique id of client

        Returns:
            int: Length of dataset
        """
        info = self.client_info[id].get()
        return int(info.detach().cpu().numpy())

    def register_net(self, net: nn.Module):
        """Register an nn.Module to server

        Args:
            net (nn.Module): the neural network
        """
        self.net = net
        self.params_size = len(pickle.dumps(self.net.state_dict()))
        self.shared_params = ConnABC(os.path.join(MMAP_PATH, SERVER_PARAMS_FILEDESC), self.params_size, 2)
        self.publish_net()

    def publish_net(self):
        """Actually publish net parameters
        The parameters are from self.net
        """
        self.shared_params.set(bundle_parameter(self.net))

    def register_client(self, id: int) -> bool:
        """Register client to server.

        Args:
            id (int): Unique id of client

        Returns:
            bool: status code
        """

        # Avoid duplicate clients
        if id in self.client_ids:
            print("[ Error ] Client already registered")
            return False

        # Complete client registration
        self.client_ids.append(id)
        self.client_params[id] = ConnABC(self.get_client_params_path(id), self.params_size)
        self.client_signal[id] = ConnABC(self.get_client_signal_path(id), 1)
        self.client_info[id] = ConnABC(self.get_client_info_path(id), len(pickle.dumps(torch.tensor(0, dtype=torch.int64))))
        self.client_signal[id].set(bytearray([self.status]), encode=False)
        self.client_info[id].set(torch.tensor(0, dtype=torch.int64))
        return True
    
    def unregister_client(self, id: int) -> bool:
        """Unregister client from server.

        Args:
            id (int): Unique id of client

        Returns:
            bool: status code
        """
        if id not in self.client_ids:
            print("[ Error ] Client not registered")
            return False
        
        self.client_ids.remove(id)
        if not self.client_params[id].closed:
            self.client_params[id].close()
        
        if not self.client_signal[id].closed:
            self.client_signal[id].close()

        if not self.client_info[id].closed:
            self.client_info[id].close()

        self.client_params.pop(id)
        self.client_signal.pop(id)
        self.client_info.pop(id)

    def publish_signal(self, signal: int=None):
        """Publish a signal, to ALL clients

        Args:
            signal ([int], optional): Signal. Defaults to None.
        """
        # Use self.status if signal is None
        if signal is not None:
            self.status = signal
        for client_id in self.client_ids:
            # Signals are raw bytes, do not encode
            self.client_signal[client_id].set(bytearray([self.status]), encode=False)
    
    def send_signal(self, id: int, signal: int):
        """Send a signal to client[id]

        Args:
            id (int): Unique id of client
            signal (int): The signal
        """
        # Signals are raw bytes, do not encode
        self.client_signal[id].set(bytearray([signal]), encode=False)

    def wait_clients(self, timeout=1e3) -> List[int]:
        """Wait for clients to complete trainning

        Args:
            timeout ([type], optional): Timeout. Defaults to 1e3.

        Returns:
            List[int]: List of accomplished clients
        """
        start_time = time.time()
        ready_clients: List[int] = []
        while time.time() < start_time + timeout:
            if len(ready_clients) == len(self.client_ids):
                break
            for client_id in self.client_ids:
                signal = self.client_signal[client_id].get(decode=False)[0]
                if signal == SIG_C_READY:
                    if client_id not in ready_clients:
                        ready_clients.append(client_id)
                        start_time = time.time()
            time.sleep(1e-1)
        return ready_clients
    
    def close(self) -> None:
        """Close server and release resources
        """
        if not self.shared_params.closed:
            self.shared_params.close()

        for client_id in self.client_ids:
            if not self.client_params[client_id].closed:
                self.client_params[client_id].close()
            if not self.client_signal[client_id].closed:
                self.client_signal[client_id].close()

    def optimize(self, ready_clients: List[int] = None) -> None:
        """Optimize self.net using parameters collected

        Args:
            clients (List[int], optional): Finished clients. Defaults to None.
        """
        # Use data from all clients
        if ready_clients is None:
            ready_clients = self.client_ids

        # Gather clien.info (client.dataset_len) and calculate total number of samples and gain for each client
        tot_samples: int = 0
        client_gain: Dict[int, float] = dict()
        for client_id in ready_clients:
            curr_info = self.get_client_info(client_id)
            tot_samples += curr_info
            client_gain[client_id] = curr_info
        print(f'[ Info ] Total number of samples: {tot_samples}')
        for key in client_gain.keys():
            client_gain[key] /= tot_samples

        # Prepare an OrderedDict for result
        new_params = OrderedDict()
        target_names: List[str] = []
        for name, _ in list(self.net._named_members(lambda module: module._parameters.items())):
            new_params[name] = 0
            target_names.append(name)

        # Gather all model parameters
        for client_id in ready_clients:
            curr_param = self.get_client_params(client_id)
            for name in new_params.keys():
                new_params[name] += curr_param[name] * client_gain[client_id]
        
        # Check if the state dict is valid
        if verify_state_dict(target_names, new_params):
            self.net.load_state_dict(new_params)
            print('[ Debug ] New params loaded')
        
        # Publish the net
        self.publish_net()

    def serve(self, n_epochs: int) -> bool:
        """Server start serving, for n epochs

        Args:
            n_epochs (int): n epochs

        Returns:
            (bool): status code
        """
        if self.net is not None:
            self.publish_signal(SIG_S_READY)
        else:
            self.publish_signal(SIG_INIT)
            return False

        for epoch_idx in range(1, n_epochs + 1):
            print(f'[ Info ] Loop {epoch_idx}')
            # Print signature of model parameters for verification
            print(f'[ Debug ] Model parameter signature: {md5(self.shared_params.get(decode=False)).hexdigest()}') 
            # print(f'[ Debug ] Parameters: {self.net.state_dict()}')
            print(f'[ Info ] Clients: {self.client_ids}')
            ready_clients: List[int] = self.wait_clients()
            if len(ready_clients) == 0:
                self.publish_signal(SIG_S_CLOSE)
                time.sleep(1)
                return False
            
            self.publish_signal(SIG_S_BUSY)
            self.optimize(ready_clients=ready_clients)
            if epoch_idx < n_epochs:
                self.publish_signal(SIG_S_READY)
            else:
                self.publish_signal(SIG_S_CLOSE)
        return True
