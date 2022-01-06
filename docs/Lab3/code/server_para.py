"""server_para.py
"""
import argparse
import torch
import torch.nn as nn
import torchvision
import tqdm
from hashlib import md5
from typing import List
import time

torch.random.manual_seed(0)

from fedpara import *
from models import ModelABC

def run(args: argparse.Namespace) -> None:
    # Init server
    server = ServerABC()
    device = torch.device(args.device)
    # Register bodel
    server.register_net(ModelABC(args).to(device).eval())

    client_list = list(range(args.world_sz))
    for client_id in client_list:
        server.register_client(client_id)

    try:
        # Serve model
        server.publish_signal(SIG_S_READY)

        for epoch_idx in range(1, args.n_epochs + 1):
            print(f'[ Info ] Loop {epoch_idx}')
            # Print signature of model parameters for verification
            print(f'[ Debug ] Model parameter signature: {md5(server.shared_params.get(decode=False)).hexdigest()}') 

            print(f'[ Info ] Clients: {server.client_ids}')
            ready_clients: List[int] = server.wait_clients()
            if len(ready_clients) == 0:
                server.publish_signal(SIG_S_CLOSE)
                time.sleep(1)
                break
            
            # Server mark its self busy
            server.publish_signal(SIG_S_BUSY)
            server.optimize(ready_clients=ready_clients)
            test(args, epoch_idx, server.net, device)
            # If this epoch is the last, close server and notify clients
            if epoch_idx < args.n_epochs:
                server.publish_signal(SIG_S_READY)
            else:
                server.publish_signal(SIG_S_CLOSE)
                break

        test(args, epoch_idx, server.net, device)
    except KeyboardInterrupt as e:
        server.close()


def test(args: argparse.Namespace, epoch_idx: int, net: nn.Module, device: torch.device = torch.device('cpu')) -> None:
    """Server tests model with the emtire dataset

    Args:
        net (nn.Module): Network
        device (torch.device, optional): Device to test model. Defaults to torch.device('cpu').
    """
    BATCH_SZ_TEST: int = 16

    net.to(device)
    net.eval()

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/',
                                                                         train=False,
                                                                         download=True,
                                                                         transform=torchvision.transforms.Compose([
                                                                             torchvision.transforms.ToTensor(),
                                                                             torchvision.transforms.Normalize((0.1307, ),
                                                                                                              (0.3081, ))
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
        
    with open(f'./logs/log_para_n:{args.world_sz}.txt', 'a+') as f:
        f.write(f'{epoch_idx},{acc_cnt / tot_cnt}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_sz', type=int)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')

    args: argparse.Namespace = parser.parse_args()
    run(args)
    # args have
    # - num_clients: int
    # - n_epochs: int default to 1
