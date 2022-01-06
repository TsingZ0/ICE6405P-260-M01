"""gen_mnist_pathological.py
"""
import torch
import random
from typing import List
import pickle
import sys, os
from fedpara import sort_mnist

if __name__ == '__main__':
    n_client: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    res_stimulis_all, res_labels_all = sort_mnist()
    n_patches: int = n_client * 2
    patch_sz = 60000 // n_patches
    data_assignment: List[int] = list(range(n_patches))
    random.shuffle(data_assignment)
    export_dir = f'./export_pathological/mnist_{n_client}'
    
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    for client_id in range(n_client):
        patch_idx_1 = data_assignment[client_id * 2]
        patch_idx_2 = data_assignment[client_id * 2  + 1]

        stimulis_tmp = torch.cat([
            res_stimulis_all[patch_idx_1 * patch_sz:(patch_idx_1 + 1) * patch_sz, ...],
            res_stimulis_all[patch_idx_2 * patch_sz:(patch_idx_2 + 1) * patch_sz, ...]
        ])
        labels_tmp = torch.cat([
            res_labels_all[patch_idx_1 * patch_sz:(patch_idx_1 + 1) * patch_sz, ...],
            res_labels_all[patch_idx_2 * patch_sz:(patch_idx_2 + 1) * patch_sz, ...]
        ]).to(torch.int64)

        data = {'stimulis': stimulis_tmp, 'labels': labels_tmp}
        with open(f'{export_dir}/client_{client_id}.pkl', 'wb') as f:
            pickle.dump(data, f)