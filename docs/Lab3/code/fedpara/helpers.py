import mmap
import pickle
import itertools
from hashlib import md5
from typing import Any, Dict, List, Union, Tuple
from collections import OrderedDict
import time
import os
import random

import torch
import torch.nn as nn
import torchvision
import tqdm

from .config import *

def sort_mnist() -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = torchvision.datasets.MNIST('./data/',
                                     train=True,
                                     download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
                                     ]))
    res_stimulis: torch.Tensor = torch.zeros(size=(10, 7000, 1, 28, 28), dtype=torch.float32)
    res_labels: torch.Tensor = torch.zeros(size=(10, 7000), dtype=torch.float32)
    res_index: torch.Tensor = torch.zeros(size=(10,),dtype=torch.int64)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    with tqdm.tqdm(dataset) as pbar:
        for item in dataset:
            label = item[1]
            res_stimulis[label, res_index[label],:,:,:] = item[0]
            res_labels[label, res_index[label]] = label
            res_index[label] += 1
            pbar.update()

    res_stimulis_all: torch.Tensor = torch.cat([res_stimulis[idx, 0:res_index[idx],...] for idx in range(10)])
    res_labels_all: torch.Tensor = torch.cat([res_labels[idx, 0:res_index[idx],...] for idx in range(10)]).to(torch.int64)

    return res_stimulis_all, res_labels_all

def break_into(n,m) -> List[List[int]]:
    """
    return m random integers with sum equal to n 
    """
    distribution = [1 for i in range(m)]

    for i in range(n-m):
        ind = random.randint(0,m-1)
        distribution[ind] += 1

    index = [i for i in range(n)]
    random.shuffle(index)

    res = [[] for i in range(m)]
    tmp: int = 0
    for idx, bin in enumerate(distribution):
        res[idx] += index[tmp: tmp + bin]
        tmp += bin

    return res

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

def gen_signature(net: Union[nn.Module, OrderedDict]) -> str:
    if isinstance(net, OrderedDict):
        return md5(pickle.dumps(net)).hexdigest()
    elif isinstance(net, nn.Module):
        return md5(pickle.dumps(net.state_dict())).hexdigest()
    else:
        raise NotImplementedError

def verify_state_dict(names: List[str], state_dict: OrderedDict) -> bool:
    for name in names:
        if name in state_dict.keys() and isinstance(state_dict[name], torch.Tensor):
            pass
        else:
            return False
    return True

class ConnABC(object):
    def __init__(self, path: str, size: int=0, mult:int=2) -> None:
        super().__init__()
        self.mult: int = mult # Multiplier of size. RealSize = Size * Multiplier
        self.path: str = path # Mapped path of shared memory
        self.size: int = size # Size of memory. When size==0, the connection will be set upon an existing file
        self.closed: bool = True
        self.open()

    def open(self):
        """Start connection
        """
        if self.closed:
            self._create_mmap()

    def _create_mmap(self) -> None:
        """Create mmap
        The server is responsible of creating mmap files. It must decide the size of share memory

        The client, on the other hand, open a mmap file directly. So size==0 on the client size, and the client should not create new file on disk
        """

        # Creating an empty file on disk
        if self.size > 0:
            with open(self.path, 'wb') as f:
                f.write(bytearray(itertools.repeat(0, int(self.size * self.mult))))
        
        # Open the file and mmap
        self.fd = open(self.path, 'r+b')
        self.mmap = mmap.mmap(self.fd.fileno(), 0, mmap.MAP_SHARED)
        self.size = self.mmap.size()
        self.closed = False
    
    
    def set(self, obj: Any, encode:bool=True) -> bool:
        """Set the content of shared memory to an object

        Args:
            obj (Any): bytes array or other types of object
            encode (bool, optional): Encode the object or not. Defaults to True.

        Raises:
            BufferError: The object exceeds size limit

        Returns:
            bool: Status
        """

        # If encode is True, encode the object with pickle
        if encode:
            obj_ser = pickle.dumps(obj)
        else:
            obj_ser = obj
        if len(obj_ser) > self.size:
            raise BufferError(f'Oversized object {len(obj_ser)} exceed limit of {self.size}')
        
        self.mmap.seek(0) # Remember to seek(0)
        self.mmap.write(obj_ser)
        return True

    def get(self, decode: bool=True) -> Any:
        """Get object from shared memory

        Args:
            decode (bool, optional): Decode the object or not. Defaults to True.

        Returns:
            Any: Result
        """
        self.mmap.seek(0) # Remember to seek(0)

        # # If decode is True, decode the object with pickle
        if decode:
            return pickle.loads(self.mmap.read())
        else:
            return self.mmap.read()

    
    def close(self):
        """Shut the connection down gracefully
        """
        if not self.mmap.closed:
            self.mmap.close()
        
        if not self.fd.closed:
            self.fd.close()
        
        self.closed = True

class ClientABC(object):
    def __init__(self, id: int, *args, **kwargs) -> None: 
        """# Client abstraction

        Args:
            id (int): client unique id
        """
        super().__init__()
        self.id: int = id
        self.status: int = None # Client status
        self.signal: ConnABC = None # Client signal, shared with server
        self.server_params: ConnABC = None # Server params, read-only
        self.client_params: ConnABC = None # Client params, shared with server, write-only
        self.client_info: ConnABC = None # Client signal, shared with server, read-write
        self.closed: bool = True

    @property
    def server_params_path(self) -> str:
        return os.path.join(MMAP_PATH, SERVER_PARAMS_FILEDESC)

    @property
    def client_params_path(self) -> str:
        return os.path.join(MMAP_PATH, CLIENT_PARAMS_FILEDESC.format(self.id))
    
    @property
    def client_info_path(self) -> str:
        return os.path.join(MMAP_PATH, CLIENT_INFO_FILEDESC.format(self.id))

    @property
    def signal_path(self) -> str:
        return os.path.join(MMAP_PATH, CLIENT_SIGNAL_FILEDESC.format(self.id))

    def set_signal(self, signal: int) -> bool:
        """Set the signal of client

        Args:
            signal (int): An integer, see fedavg_config.py

        Returns:
            bool: Status code
        """
        self.status = signal
        # Raw bytes used for signal, do not encode
        self.signal.set(bytearray([signal]), encode=False)
        return True
    
    def get_signal(self) -> int:
        """Get the signal

        Returns:
            int: signal
        """
        # Raw bytes used for signal, do not decode
        return self.signal.get(decode=False)[0]

    def get_params(self) -> OrderedDict:
        """pull params form server

        Returns:
            OrderedDict: state_dict
        """
        params = self.server_params.get()
        return params
    
    def set_params(self, model: nn.Module) -> bool:
        """Push params to shared memory

        Args:
            model (nn.Module): The current model

        Returns:
            bool: Status code
        
        Warning:
            Use bundle_parameter and copy tensors to CPU
        """
        self.client_params.set(bundle_parameter(model))
        return True
    
    def set_info(self, info: int) -> bool:
        """Set client info (length of dataset)

        Args:
            info (int): The length is an integer

        Returns:
            bool: Status code
        """
        # Warning: pickle does not dump int to a fixed length bytearray, therefore, the length must be converted to torch.tensor
        self.client_info.set(torch.tensor(info, dtype=torch.int64))
        return True

    def open(self) -> None:
        self.client_params = ConnABC(self.client_params_path, 0)
        self.signal = ConnABC(self.signal_path, 0)
        self.server_params = ConnABC(self.server_params_path, 0)
        self.client_info = ConnABC(self.client_info_path, 0)

    def close(self) -> None:
        if not self.client_params.closed:
            self.client_params.close()
        if not self.signal.closed:
            self.signal.close()
        if not self.server_params.closed:
            self.server_params.close()
        if not self.client_info.closed:
            self.client_info.close()
        
        self.closed = True
    
    def wait_server(self) -> int:
        """Endless loop that checks signal from server

        Returns:
            int: signal obtained
        """
        while True:
            signal = self.get_signal()
            if signal == SIG_S_READY or signal == SIG_S_CLOSE or signal == SIG_S_CLOSE:
                return signal

            # time.sleep to avoid high CPU consumption
            time.sleep(1e-1)

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
