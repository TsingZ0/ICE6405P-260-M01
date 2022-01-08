# README

## Summary

The `dataset_dl` folder is a demostration of how dataset is organized. You should run the `../scripts/ownload_dataset.py` to download MNIST data via torchivision and put it to right place

The `datset_deploy` folder stores packed datasets

## How to build

```console
docker build .
```

The `worker_utils.py` `run_worker.py` `models.py` are neccessary to build workder

## Files

### file_uploader.py

Upload resources to OSS

### models.py

Defines LeNet5 model

### run_server.py

Start a parameter server at port 29500

> The script is only compatible with unix. It does not support Darwin systems (MacOS)

### run_worker.py

Start a worker. The worker accept two api,  `:8080/init` and `:8080/run`.

### test_accuracy.py

Test accuracy of a model parameter from parameter server

### worker_utils.py

Helper functions for worker
