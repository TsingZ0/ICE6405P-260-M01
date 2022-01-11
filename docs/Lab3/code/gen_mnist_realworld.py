"""gen_realworld_pathological.py
"""
import torch
import pickle
import sys, os
from fedpara import sort_mnist, break_into

if __name__ == '__main__':
    n_client: int = int(sys.argv[1]) if len(sys.argv) > 0 else 10

    res_stimulis_all, res_labels_all = sort_mnist()
    n_patches = 5 * n_client
    patch_sz = 60000 // n_patches
    data_assignment = break_into(n_patches, n_client)
    export_dir = f'./export_realworld/mnist_{n_client}'

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    for client_id in range(n_client):
        stimulis_tmp = torch.cat([
            res_stimulis_all[patch_idx * patch_sz:(patch_idx + 1) * patch_sz, ...] for patch_idx in data_assignment[client_id]
        ])
        labels_tmp = torch.cat([
            res_labels_all[patch_idx * patch_sz:(patch_idx + 1) * patch_sz, ...] for patch_idx in data_assignment[client_id]
        ]).to(torch.int64)
        data = {'stimulis': stimulis_tmp, 'labels': labels_tmp}
        with open(f'{export_dir}/client_{client_id}.pkl', 'wb') as f:
            pickle.dump(data, f)
