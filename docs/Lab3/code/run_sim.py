"""run_sim.py
"""
from typing import Tuple, Dict
import argparse
import logging
import faulthandler
from concurrent.futures import ThreadPoolExecutor, as_completed

faulthandler.enable()

import torch
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST

torch.random.manual_seed(0)

from mnist_noniid_dataset import MNISTNonIID
from models import LeNet5
from fedsim import ClientSimBackend, ServerSim, ClientSim, partition


def train_async(client: ClientSim, server_params: Dict[str, torch.Tensor],
                dataset: Dataset) -> Tuple[float, Dict[str, torch.Tensor]]:
    logging.info(f'Training client {client.id}')
    client.new_parameters = server_params
    # Each client train its parameter with local data
    client(dataset)
    # Server collects new parameter from clients
    return len(client), client.parameters


def main(args: argparse.Namespace):
    # Initialize server
    server = ServerSim(LeNet5(), device=torch.device(args.s_device))
    # Parse devicese.g. --c_device=cuda:0,cuda:1 -> ['cuda:0','cuda:1']
    c_devices = [torch.device(dev_str) for dev_str in args.c_device.split(',')]
    # Creating client backends. They are executer of clients
    client_backends = [ClientSimBackend(id=idx, net=server.net, device=c_devices[idx]) for idx in range(len(c_devices))]
    # Creating clients. Each client is assigned to a backend
    clients = [
        ClientSim(id=idx,
                  backend=client_backends[idx % len(client_backends)],
                  n_epochs=args.n_epochs,
                  batch_sz=args.batch_sz,
                  lr=args.lr,
                  criterion=torch.nn.functional.cross_entropy,
                  optim=torch.optim.Adam) for idx in range(args.world_sz)
    ]
        # Initialize dataset classes
    client_datasets = [
        MNISTNonIID(f'./export_{args.dataset_type}/mnist_{args.world_sz}/client_{idx}.pkl', device=clients[idx].device) for idx in range(args.world_sz)
    ]

    n_threads = len(client_backends)

    # Loop for n_sim times
    for sim_idx in range(1, args.n_sim + 1):
        # Temporarily cache server parameters
        server_params = server.parameters
        with tqdm.tqdm(range(args.world_sz), nrows=2) as pbar:
            # Slice clients and cilent_datasets to len(client_backends)
            for batch_clients, batch_dataset in zip(partition(clients, n_threads),
                                                    partition(client_datasets, n_threads)):
                # Each client train seperately using threadpool
                executor = ThreadPoolExecutor(max_workers=len(client_backends))
                tasks = [
                    executor.submit(train_async,
                        batch_clients[idx],
                        server_params,
                        batch_dataset[idx],
                    ) for idx in range(len(batch_dataset))
                ]
                for future in as_completed(tasks):
                    res = future.result()
                    server[res[0]] = res[1]
                pbar.set_description(f'sim: {sim_idx}, client: {id}, world_sz:{args.world_sz}')
                pbar.update()
        # Optimize server parameters based on collected parameters
        server()
        test(str(args.world_sz), sim_idx, server.net, server.device)


def test(text: str, epoch_idx: int, net: nn.Module, device: torch.device = torch.device('cpu')) -> None:
    """Server tests model with the emtire dataset

    Args:
        net (nn.Module): Network
        device (torch.device, optional): Device to test model. Defaults to torch.device('cpu').
    """
    BATCH_SZ_TEST: int = 16

    net.to(device)
    net.eval()

    test_loader = DataLoader(MNIST('./data/',
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
                                   ])),
                             batch_size=BATCH_SZ_TEST,
                             shuffle=True)

    acc_cnt: int = 0
    tot_cnt: int = 1e-5

    with tqdm.tqdm(range(len(test_loader))) as pbar:
        for batch_idx, (stimulis, label) in enumerate(test_loader):
            pred = net(stimulis.to(device))
            pred_decoded = torch.argmax(pred, dim=1)
            acc_cnt += (pred_decoded == label.to(device)).sum().detach().cpu().numpy()
            tot_cnt += pred_decoded.size(0)
            pbar.set_description("acc:{}".format(acc_cnt / tot_cnt))
            pbar.update(1)

    with open(f'./logs/log_sim_n:{text}.txt', 'a+') as f:
        f.write(f'{epoch_idx},{acc_cnt / tot_cnt}\n')


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_sz', type=int, help='Number of clients')
    parser.add_argument('--n_sim', type=int, default=1, help='Number of simulation loops')
    parser.add_argument('--s_device', type=str, default='cpu', help='Device to put server parameters')

    parser.add_argument('--n_epochs', type=int, default=1, help='Number of loops on client side')
    parser.add_argument('--dataset_type', type=str, default='pathological', help='pathological | realworld')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_sz', type=int, default=32, help="Batch size")
    parser.add_argument('--c_device', type=str, default='cpu', help="Device to put client parameters")
    parser.add_argument('--n_multiplex', type=int, default=1, help="Reuse gpu for multiple client")

    args = parser.parse_args()
    # Run main with asyncio
    main(args)
