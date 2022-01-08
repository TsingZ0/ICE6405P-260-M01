# Serverless ML App

## How to train

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

## How to build and deploy

Only `deploy-flask.py`, `model.onnx` are needed for deployment. See the Dockerfile for details. But in general `docker build .` should work.

## Files

### requirements.txt

Packages required to run this project

### train.py

Script to train the network

### model.py

Defines a LeNet5 network

### export_onnex.py

Export a trained LeNet45 model to onnx format
### deploy-flask.py

This is the main app. To run the app, execute

```console
python deploy-flask.py
```
