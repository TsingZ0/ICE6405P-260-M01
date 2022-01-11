# FedAvg Experiment

## Before experiment

Place data set to `./data`

```text
./data
└── MNIST
    ├── processed
    │   ├── test.pt
    │   └── training.pt
    └── raw
        ├── t10k-images-idx3-ubyte
        ├── t10k-labels-idx1-ubyte
        ├── train-images-idx3-ubyte
        └── train-labels-idx1-ubyte
```

## Common files

`gen_mnist_pathological.py` can generate pathological dataset
`gen_mnist_realworld.py` can generate realworld dataset

The two script depend on `fedpara` package, `torch`, `torchvision` and other Python packages. You should create `./export_realworld` and `./export_pathological` in advance.

After generation, the serialized MNIST data will be stored in corresponding folder.

`mnist_noiid_dataset.py` contains a definition of `MNISTNonIID`, which is designed for loading both pathological and realworld dataset. It use `pickle` to load previously generated MNIST data

`models.py` contains the definition of LeNet5 model.

## For FedAvg simulator

`run_sim.py` and `fedsim` package is important. `run_sim.sh` is just a script to automate experiment.

## For FedAvg Parallel simulator

`server_para.py` will start the local paramter server that shares model parameter via shared memory according to command-line parameters.
`client_para.py` will start the client according to command-line parameters. `start_clients.py` offers an easy way to start multiple client simutaneously using `subprocess`.

## Other Files

`visualize_mnist_realworld.ipynb` to explore the distribution of realworld dataset
`visualize_data` to explore the training result.