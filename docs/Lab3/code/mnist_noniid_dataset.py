"""mnist_noniid_dataset.py
"""
import torch
from torch.utils.data import Dataset
import pickle
from typing import Any

class MNISTNonIID(Dataset):
    stimulis: torch.Tensor = None
    labels: torch.Tensor = None
    length: int = 0
    def __init__(self, path_to_pkl: str, device: torch.device=torch.device('cpu')) -> None:
        super().__init__()
        with open(path_to_pkl, 'rb') as f:
            data = pickle.load(f)

        self.stimulis = data['stimulis'].to(device)
        self.labels = data['labels'].to(device)
        self.length = len(self.stimulis)
    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> Any:
        return self.stimulis[index,...], self.labels[index]
