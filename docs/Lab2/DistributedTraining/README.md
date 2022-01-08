# Distributed Training

- [Distributed Training](#distributed-training)
  - [Preface](#preface)
  - [Project Design](#project-design)
    - [The Parameter Server](#the-parameter-server)
    - [The Docker Contained Worker](#the-docker-contained-worker)
  - [Preparing Dataset](#preparing-dataset)
    - [Explanation](#explanation)
    - [Summary of urls](#summary-of-urls)
  - [Training](#training)
    - [API test with curl](#api-test-with-curl)
  - [Build Docker Image](#build-docker-image)
  - [Configure OpenWhisk](#configure-openwhisk)
  - [Changing server configuration](#changing-server-configuration)
  - [Summary](#summary)

## Preface

> 这是一个示例应用程序 Dark vision，它就是这样做的。在此应用程序中，用户使用 Dark Vision Web 应用程序上载视频或图像，该应用程序将其存储在 Cloudant DB 中。视频上传后，OpenWhisk 通过听 Cloudant 更改（触发）来检测新视频。然后，OpenWhisk 触发视频提取器操作。在执行过程中，提取器将生成帧（图像）并将其存储在 Cloudant 中。然后使用 Watson Visual Recognition 处理帧，并将结果存储在同一 Cloudant DB 中。可以使用 Dark Vision Web 应用程序或 iOS 应用程序查看结果。除 Cloudant 外，还可以使用对象存储。这样做时，视频和图像元数据存储在 Cloudant 中，媒体文件存储在对象存储中。
>
> _from [【无服务器架构】openwhisk 经典使用案例](https://www.163.com/dy/article/GBJMDQNT0511DQI7.html)_

## Project Design

We intend to use the OpenWhisk serviceless computing framework for distributed machine learning. In order to do this, we first need to set up a parameter server for saving model parameters. This server provides  parameters for worker. This server can be implemented on a cluster for maximum performance, but given the simplicity of the experiment, we deploy it to a single machine.

Second, we need to define a serviceless function for training:

$$
\mathtt{NewParameters} = Train(\mathtt{Model}, \mathtt{Parameters}, \mathtt{Dataset})
$$

This function accepts the model, the model parameters, and the dataset as input and outputs the updated parameters. Considering the limitations of the OpenWhisk framework, in practice we fix the model to `LeNet5`, locate the dataset with a URL and pull the parameters from an external parameter server, hence the following pseudo code.

```text
def Train(Model=LeNet5, ParameterServerURL, DatasetURL, TrainSettings):
    Dataset = GetFromURL(DatasetURL)
    Parameters = PullFromParameterServer(ParameterServerURL)
    TrainWithModel(Model, Parameters, Dataset, TrainSettings)
```

### The Parameter Server

A parameter server is created by [`Flask`](https://flask.palletsprojects.com/en/2.0.x/) to serve model parameters to workers. Its mission is to:

- Store model parameters in `model`, keep it versioned by updating `model.version`
- Respond to workers' request of latest model parameters via `/getParameter`
- Acceput gradient uploaded from workers from `/putGradient`
- Optimize model parameters according to gradient received

The api of this server is summarized as bellow:

| API                 | Method | Type                     | Example                                                            |
| ------------------- | ------ | ------------------------ | ------------------------------------------------------------------ |
| `/getVersion`       | GET    | application/json         | {"code":200, "accuracy":0.0}                                       |
| `/getParameter`     | GET    | application/octec_stream | serialized state dict {"code":200,"param":Dict[str, torch.Tensor]} |
| `/putGradient`      | POST   | application/octec_stream | serialized gradient dict {"id":0,"param":Dict[str, torch.Tensor]}  |
| `/registerWorker`   | POST   | application/json         | (beta){ "id":0,"description":"worker_0"}                           |
| `/unregisterWorker` | POST   | application/json         | (beta){ "id":0}                                                    |

> `/registerWorker` and `/unregisterWorker` is under development. The interface should be protected by
> some sort of access/secret key pair and SSL encryption to avoid abuse of model parameters. On the other hand,
> sending pickle serialized object over http is not secure. We are aware of these vulnerabilities

Once updated, `model.version` will increase. The workers can check version of model via `/getVersion` api. If the model is updated on the parameter server, workers can choose to download latest parameters to local

Full code of parameter server can be found in `run_server.py`

### The Docker Contained Worker

The worker will be launched by Openwhisk. Therefore, we need to create: two api `/init` and `/run`

| API     | Method | Type             | Significance      |
| ------- | ------ | ---------------- | ----------------- |
| `/init` | POST   | None             | Init NeuralNet    |
| `/run`  | POST   | application/json | Execute trainning |

Since we are building our own python action, `/init` api is insignificant:

```python
@app.route("/init", methods=['GET', 'POST'])
def init():
    global g_net, g_train_dataset
    g_net = Net()
    g_net.train()
```

The `/run` api, on the other hand, need parameters that are critical to training. After experiments, we decideded that `/run` api will accept a json dictionary like this:

```json
{
    "value": {
        "batch_sz_train": 32,
        "epoch_n": 32,
        "apihost": "http://192.168.1.131:29500",
        "update_intv": 8,
        "dataset_url": "http://192.168.1.131:9000/mnist-dataset/dataset_dl_1.tar.gz",
        "device": "cpu"
    }
}
```

Where the `value` key is **automatically added by OpenWhisk**.

Full code of worker can be found  in `run_worker.py` and `worker_utils.py`

##  Preparing Dataset

The goal of this section is to make datasets accessible via URL

First, we split mnist dataset into 6 parts. A `dataset_dl` module is organized as follows :

```text
./dataset_dl
├── __dataset__.py
├── __init__.py
└── data
    └── MNIST
        └── raw
            ├── t10k-images-idx3-ubyte
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte
            ├── train-images-idx3-ubyte.gz
            ├── train-labels-idx1-ubyte
            └── train-labels-idx1-ubyte.gz
```

### Explanation

The `__init__.py` import dataset object from`__dataset__.py`

```python
"""__init__.py"""
from .__dataset__ import dataset as Dataset
```

In `__dataset__.py`, the full MNIST dataset is truncated by index. The example below shows a subset of MNIST from 50000 to 60000 (10000 elements)

```python
"""__dataset__.py"""
import torch
import torchvision

_raw_dataset = torchvision.datasets.MNIST(
        './dataset_dl/data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
# Split dataset using torch.utils.data.Subset api
dataset = torch.utils.data.Subset(_raw_dataset, range(50000, 60000))
```

> We admit this format as a paradigm. Any dataset can be wrapped and accessed by:
>
> ```python
> import dataset_dl
> dataset = dataset_dl.Dataset # a torch.utils.data.Datasset object
> ```
>
> without knowiing the detail of implementation

We then create 6 dataset, each dataset is a different subset of the whole MNIST dataset

```bash
$ tar -zcvf dataset_dl_1.tar.gz ./dataset_dl # Subset(0,10000)
$ tar -zcvf dataset_dl_2.tar.gz ./dataset_dl # Subset(10000,20000)
$ tar -zcvf dataset_dl_3.tar.gz ./dataset_dl # Subset(20000,30000)
$ tar -zcvf dataset_dl_4.tar.gz ./dataset_dl # Subset(30000,40000)
$ tar -zcvf dataset_dl_5.tar.gz ./dataset_dl # Subset(40000,50000)
$ tar -zcvf dataset_dl_6.tar.gz ./dataset_dl # Subset(50000,60000)
```

We upload these datasets to OSS with the help of previously created`file_uploader.py`, or via web interface of MinIO.

```bash
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_1.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'd96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/d96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090622Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e44304c335bbd5ef00c8726eef207d8b3448956e35b072d60a5f4d4af08b0987'}

```

### Summary of urls

Finally, we have a list of dataset that can be accessed via URL.
| Volume | URL                                                                                                                        |
| ------ | -------------------------------------------------------------------------------------------------------------------------- |
| 1      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_1.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_1.tar.gz) |
| 2      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_2.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_2.tar.gz) |
| 3      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_3.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_3.tar.gz) |
| 4      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_4.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_4.tar.gz) |
| 5      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_5.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_5.tar.gz) |
| 6      | [http://192.168.1.131:9000/mnist-dataset/dataset_dl_6.tar.gz](http://192.168.1.131:9000/mnist-dataset/dataset_dl_6.tar.gz) |

> The dataset on OSS (data and `.py` descriptors) must be tar.gz archive.

## Training

### API test with curl

Before deployment, we first test distributed workers with `curl`.

We first fire up the MinIO instance with `docker start minio`. Then the parameter server via

```console
$ python run_server.py
```

The parameter server will listen at port 29500

![run_servers](./img/2022-01-08-16-34-40.png)

Then, we start worker via:

```console
$ python run_worker.py
```

The `run_worker.py` will listen at port 8080. We need to use this API via `curl`.

```bash
#!/bin/bash
SERVER_HOST=http://172.17.0.1:29500
DATASET_URL=http://172.17.0.1:9000/mnist-dataset/dataset_dl_1.tar.gz
WORKER_HOST=http://localhost:8080
curl -X POST \
     -d '{"value":{"batch_sz_train": 32, "epoch_n": 32, "apihost": "'$SERVER_HOST'","update_intv": 8, "dataset_url": "'$DATASET_URL'","device": "cpu"}}' \
     -H 'Content-Type: application/json' $WORKER_HOST/run
```

> See `test-worker.sh`

The training loop should start. And we can observe client activities from server side

## Build Docker Image

After local test is completed. We construct the docker image in which the worker runs

```bash
$ cd ./code # root of code
$ docker build . -t python3action-dist-train-mnist
```

After the build, we test the containner locally with curl

```bash
$ docker run --rm --net=host python3action-dist-train-mnist
$ curl ...
```

This will create a docker container that use host network. We test this container with the same method.After test, we tag the image and push it to docker registery

```bash
$ docker login
$ docker tag python3action-dist-train-mnist natrium233/python3action-dist-train-mnist:1.0
$ docker push natrium233/python3action-dist-train-mnist:1.0  
```

## Configure OpenWhisk

Create OpenWhisk action

```bash
$ wsk action create dist-train --docker natrium233/python3action-dist-train-mnist:1.0 --web true --timeout 300000
```

> This command assumes that openwhisk is configured with api-gateway, otherwise you should remove the `--web true` flag

To invoke this action, run

```bash
#!/bin/bash
SERVER_HOST=http://172.17.0.1:29500 
DATASET_URL=http://172.17.0.1:9000/mnist-dataset/dataset_dl_1.tar.gz
wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost $SERVER_HOST \
    --param update_intv 8 \
    --param dataset_url $DATASET_URL \
    --param device cpu
```

The action will create workers, and we can monitor worker activity via parameter server.

> Modify SERVER_HOST and DATASET_URL on demand

> See `test-wsk-action.sh` for details

## Changing server configuration

When setting local batch size to 32, the memory consumption is 166MiB, which does not exceed the default memory limit of OpenWhisk

```bash
$ docker stats
CONTAINER ID   NAME                      CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O        PIDS
5ceb5ca66318   wsk0_13_guest_disttrain   57.77%    165.8MiB / 256MiB     64.76%    33.7MB / 35.4MB   5.93MB / 55MB    4
560ba865b754   wsk0_9_prewarm_nodejs14   0.00%     10.16MiB / 256MiB     3.97%     4.08kB / 0B       115kB / 0B       8
```

When invoking two actions at same time, multiple containers will be created:

```bash
CONTAINER ID   NAME                       CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O        PIDS
c35b1fee108c   wsk0_14_guest_disttrain    33.04%    166.5MiB / 256MiB     65.04%    30.6MB / 13.8MB   2.97MB / 55MB    4
9cb88380585d   wsk0_15_guest_disttrain    33.64%    167.1MiB / 256MiB     65.28%    30.5MB / 13.6MB   3.6MB / 55MB     4
15e384116b43   wsk0_16_prewarm_nodejs14   0.00%     10.39MiB / 256MiB     4.06%     3.28kB / 0B       401kB / 0B       8
```

When creating action, you can use `--memory` to limit the maximum memory a function can use, this value is `256MB` by default

```bash
wsk action create dist-train --docker natrium233/python3action-dist-train-mnist:1.0 --web true --timeout 300000 --memory 512
```

## Summary

The 4GB virtual machine I created have 2.5GiB of free memory at idle stat. But since each activation consumes 33% of CPU, the virtual machine can only hold up to 3 workers at one time

If we insist to add workers on this machine, the trainning will be bottenecked by CPU

If we could deploy OpenWhisk to a Kubernetes cluster, the number of workers can increase. What is more, if the cluster had GPU installed and `nvidia-docker` installed, the traininig would be accelerated by GPU.

However, configuring `nvidia-docker` on a Kubernetes cluster is tredious.

Pay attention to OpenWhisk timeout policy. On a standalone server, an action must finish within its timeout (which is 300000 milliseconds maximum).
