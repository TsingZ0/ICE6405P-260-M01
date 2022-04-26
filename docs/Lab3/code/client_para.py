"""client_para.py
"""
import argparse
from hashlib import md5
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import tqdm

from fedpara import *
from models import ModelABC
from mnist_noniid_dataset import MNISTNonIID

net: nn.Module = None
optimizer: Optimizer = None
criterion: nn.Module = None
dataset: Dataset = None
train_loader: DataLoader = None


def train(args: argparse.Namespace, client: ClientABC) -> None:
    """Basic train loop

    Args:
        args (argparse.Namespace): client arguments
        client (ClientABC): client abstraction
    """
    global net, dataset, optimizer, criterion, train_loader
    net.train()
    device = torch.device(args.device)
    for epoch_idx in range(1, args.n_epochs + 1):
        with tqdm.tqdm(range(len(train_loader))) as pbar:
            for stimuli, label in train_loader:
                optimizer.zero_grad()
                pred = net(stimuli.to(device))
                loss = criterion(pred, label.to(device))
                loss.backward()
                optimizer.step()
                pbar.set_description(f'[ Info ][id:{client.id}] loop={epoch_idx}, loss={loss.detach().cpu().numpy()}')
                pbar.update()

    client.set_info(len(dataset))
    client.set_params(net)
        

def init(args) -> None:
    n_cpu = args.num_cpu
    os.environ ['OMP_NUM_THREADS'] = str(n_cpu)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(n_cpu)
    os.environ ['MKL_NUM_THREADS'] = str(n_cpu)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(n_cpu)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(n_cpu)
    torch.set_num_threads(n_cpu)

    """Initialize model, optimizer, criterion, dataloader

    Args:
        args ([type]): [description]
    """
    global net, optimizer, criterion, dataset, train_loader

    net = ModelABC()
    device = torch.device(args.device)
    net.to(device)
    dataset = MNISTNonIID(f'./export_{args.dataset_type}/mnist_{args.world_sz}/client_{args.id}.pkl')
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    criterion = torch.nn.functional.cross_entropy
    train_loader = DataLoader(dataset, batch_size=args.batch_sz, shuffle=True)


def run(args: argparse.Namespace) -> None:
    """Run client

    Args:
        args (argparse.Namespace): Start arguments
    """
    global net

    client = ClientABC(args.id)
    client.open()
    client.wait_server()

    while True:
        params = client.get_params()
        print(f'[ Debug ][id:{client.id}] Server parameter signature: {md5(client.server_params.get(decode=False)).hexdigest()}')
        net.load_state_dict(params)

        client.set_signal(SIG_C_BUSY)
        train(args, client)
        time.sleep(1e-2)
        client.set_signal(SIG_C_READY)
        
        signal = client.wait_server()
        if signal == SIG_S_ERROR or signal == SIG_S_CLOSE:
            client.close()
            print(f'[ Info ][id:{client.id}] Client shutdown.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--dataset_type', type=str, default='pathological')
    parser.add_argument('--world_sz', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_cpu', type=int, default=4)

    args: argparse.Namespace = parser.parse_args()
    # args have
    # - id: unique id of client, start from 0
    # - n_epochs: local epoc
    # - dataset_type
    # - world_sz: int total number of clients
    # - lr
    # - batch_sz
    init(args)
    run(args)
