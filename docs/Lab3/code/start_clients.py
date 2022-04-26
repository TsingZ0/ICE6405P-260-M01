import subprocess
import time
from typing import List
import argparse
import torch
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='client_para.py')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--dataset_type', type=str, default='pathological')
    parser.add_argument('--world_sz', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')

    args: argparse.Namespace = parser.parse_args()
    # args have
    # - id: unique id of client, start from 0
    # - n_epochs: local epoc
    # - dataset_type
    # - world_sz: int total number of clients
    # - lr
    # - batch_sz
    device_str: str = args.device # cuda:0,cuda:1
    devices = device_str.split(',') #['cuda:0','cuda:1']


    process_list = []
    for id in range(args.world_sz):
        # Dispatch tasks evenlly to GPUs
        p = subprocess.Popen(f'{sys.executable} {args.name} --id={id} --n_epochs={args.n_epochs} --lr={args.lr} --device={devices[id % len(devices)]} --batch_sz={args.batch_sz} --dataset_type={args.dataset_type} --world_sz={args.world_sz}', shell=True)
        process_list.append(p)
        time.sleep(0.5)
        print(f'[ Info ] Launching Client {id} on device: {devices[id % len(devices)]}')
    
    print(f"[ Info ] Processes: {process_list}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in process_list:
            try:
                p.kill()
                p.terminate()
            except:
                print(f"[ Error ] Unable to kill processes")
        raise KeyboardInterrupt