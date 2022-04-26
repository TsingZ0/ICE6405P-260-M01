from collections import OrderedDict
import time
import os

import torch
import torch.nn as nn

from .conn import ConnABC
from .config import *
from .helpers import bundle_parameter

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