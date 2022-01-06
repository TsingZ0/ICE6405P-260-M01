# Distributed Training

> 这是一个示例应用程序 Dark vision，它就是这样做的。在此应用程序中，用户使用 Dark Vision Web 应用程序上载视频或图像，该应用程序将其存储在 Cloudant DB 中。视频上传后，OpenWhisk 通过听 Cloudant 更改（触发）来检测新视频。然后，OpenWhisk 触发视频提取器操作。在执行过程中，提取器将生成帧（图像）并将其存储在 Cloudant 中。然后使用 Watson Visual Recognition 处理帧，并将结果存储在同一 Cloudant DB 中。可以使用 Dark Vision Web 应用程序或 iOS 应用程序查看结果。除 Cloudant 外，还可以使用对象存储。这样做时，视频和图像元数据存储在 Cloudant 中，媒体文件存储在对象存储中。
>
> _from [【无服务器架构】openwhisk 经典使用案例](https://www.163.com/dy/article/GBJMDQNT0511DQI7.html)_

## Designing the Parameter Server

A parameter server is created with `Flask` to serve model parameters to workers. Its jod is to:

- Store model parameters, keep it versioned
- Respond to workers' request of latest model parameters
- Acceput gradient uploaded from workers
- Optimize model parameters according to gradient received

Once updated, `model.version` will increase. The workers can check version of model via `/getVersion` api. If the model is updated on the parameter server, workers can choose to download latest parameters to local

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

Full code of parameter server:

```python
import argparse
import os
import multiprocessing as mp
from threading import Lock
import pickle
import json
import queue
import random

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np


from gevent import pywsgi
from flask import Flask, request, Response
from typing import Dict, Any, Union


from models import Net

RANDOM_SEED=0
torch.manual_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

app = Flask(__name__)

g_net: Union[None, nn.Module] = None
g_lock: Lock = Lock()
g_workers: Dict[Any, str] = None
g_gradient_queue: mp.Queue = None
g_optimizer: Optimizer = None
g_policy: Dict[str, Any] = None
g_version: int = None

@app.route("/registerWorker", methods=['POST'])
def register_worker() -> Response:
    """Get worker registration info, and register worker

    Worker send json formated registration info:
    {
        "id":$WORKER_UNIQUE_ID
        "description":$WORKER_DESCRIPTION
    }
    Returns:
        Response: json response of status
    
    TODO: 
        use access_key and secret_key to protect this interface
    """
    resp = Response(mimetype='application/json')
    try:
        worker_info: Dict[str, Any] = request.get_json()
    except KeyError:
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if "id" not in worker_info.keys():
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if worker_info["id"] in g_workers.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "The worker has already registered"})
        return resp

    else:
        if "description" in worker_info.keys():
            g_workers[worker_info["id"]] = str(worker_info["description"])
        else:
            g_workers[worker_info["id"]] = ''

        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "Successfully registered worker"})
        return resp


@app.route("/unregisterWorker", methods=['POST'])
def unregister_worker() -> Response:
    """Unregister a worker

    Worker send json formated registration info:
    {
        "id":$WORKER_UNIQUE_ID
    }
    Returns:
        Response: json response of status
    
    TODO: 
        use access_key and secret_key to protect this interface
    """
    resp = Response(mimetype='application/json')
    try:
        worker_info: Dict[str, Any] = request.get_json()
    except KeyError:
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if "id" not in worker_info.keys():
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    if worker_info["id"] not in g_workers.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "The worker has not registered"})
        return resp

    else:
        del g_workers[worker_info["id"]]
        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "Successfully unregistered worker"})
        return resp


@app.route("/getAccuracy", methods=['GET'])
def get_accuracy():
    # TODO: Implement this feature
    resp = Response(status=200, mimetype='application/json')
    resp.data = json.dumps({"code": 200, "accuracy": 0.0})
    return resp

@app.route("/getVersion", methods=['GET'])
def get_version():
    resp = Response(status=200, mimetype='application/json')
    resp.data = json.dumps({"code": 200, "version": g_version})
    return resp

@app.route("/getParameter", methods=['GET'])
def get_parameter() -> Response:
    """Send parameter to workers

    Returns:
        Response: octet-stream response of serialized json
    
    TODO:
        use encryption to protect model
    """
    global g_lock
    resp = Response(mimetype='application/octet-stream')

    if Net is None:
        resp.data = 500
        resp.data = pickle.dumps({"code": 500, "param": ''})
        return resp

    with g_lock:
        state_dict = g_net.state_dict()
    state_dict_cpu = {}
    for key in state_dict.keys():
        state_dict_cpu[key] = state_dict[key].detach().cpu()

    resp.status = 200
    resp.data = pickle.dumps({"code": 200, "param": state_dict_cpu})
    return resp


@app.route("/putGradient", methods=['POST'])  # app.route does not accept POST actions by default
def put_gradient() -> Response:
    """Get gradient form workers
    Workers post octec_stream data of a serialized dict:
    {
        "id":$WORKER_UNIQUE_ID
        "param":state_dict
    }

    Returns:
        Response: json response of status
    
    TODO:
        Verify id, protect with access_key and secret_key
    """
    global g_gradient_queue, g_policy

    resp = Response(mimetype='application/json')
    data = request.get_data()
    if request.mimetype != 'application/octet-stream':
        resp.status = 400
        resp.data = json.dumps({"code": 400, "msg": "Bad Request"})
        return resp

    data_dict = pickle.loads(data)
    if not isinstance(data_dict, dict) or "param" not in data_dict.keys():
        resp.status = 409
        resp.data = json.dumps({"code": 409, "msg": "Wrong format"})
        return resp

    gradient_dict: Dict[str, torch.Tensor] = data_dict["param"]
    try:
        g_gradient_queue.put(gradient_dict, block=True, timeout=5)
    except queue.Full:
        resp.status = 503
        resp.data = json.dumps({"code": 503, "msg": "Too busy"})
        return resp

    with g_lock:
        ret: int = run_gradient_descent(g_policy)
    if ret == 500:
        resp.status = 500
        resp.data = json.dumps({"code": 500, "msg": "Error"})
        return resp
    elif ret == 201:
        # Gradient stored, but with no update
        resp.status = 201
        resp.data = json.dumps({"code": 201, "msg": "Error"})
        return resp
    else:
        # A parameter update is triggered
        resp.status = 200
        resp.data = json.dumps({"code": 200, "msg": "OK"})
        return resp


def apply_grad(net: nn.Module, grad_dict: Dict[str, torch.Tensor], gain: float=1.0):
    """Apply gradient to a module

    net.parameters.grad += grad_dict * gain
    Args:
        net (nn.Module): The module to apply on
        grad_dict (Dict[str, torch.Tensor]): The gradient
    """
    net.zero_grad()
    for name, param in net.named_parameters():
        try:
            if param.grad is None:
                param.grad = grad_dict[name] * gain
            else:
                param.grad += grad_dict[name] * gain
        except KeyError as err:
            print(f'[ Warning ] Key {name} does not exist')


def run_gradient_descent(policy: Dict[str, Any]) -> int:
    """Run gradient descent forever

    Args:
        policy (Dict[str, Any]): Defines strategy of train, should at least contains:
        - batch_sz [int]: Gradient descent batch size
        - lr [float]: learning rate
        - save_interval
    """
    global g_gradient_queue, g_net, g_optimizer, g_version
    batch_sz = g_policy["batch_sz"]
    cnt = 0
    grad_list = []
    # If collected enough gradient, run optimizer
    if (g_gradient_queue.qsize() > g_policy["batch_sz"]):
        try:
            while cnt < batch_sz:
                grad_list.append(g_gradient_queue.get_nowait())
                cnt += 1
        except queue.Empty:
            print("Empty Queue")
            return 500
        
        gain = 1 / len(grad_list)
        for curr_grad in grad_list:
            apply_grad(g_net, curr_grad, gain)
        
        print("[ Info ] Optimizer step")
        g_optimizer.step()
        g_optimizer.zero_grad()
        g_version += 1

        return 200
    else:
        return 201
 
def _init_share(queue_maxsize: int=4096):
    """Init share variables

    Args:
        queue_maxsize (int, optional): Shared queue that stores collected gradients. Defaults to 4096.
    """
    global g_gradient_queue
    global g_workers

    g_workers = dict()
    g_gradient_queue = mp.Queue(maxsize=queue_maxsize)

def _init_net(model: nn.Module, 
              pth_to_state_dict: str = None, 
              device: torch.device = torch.device('cpu'), *args, **kwargs):
    """Init neural network

    Args:
        model (nn.Module): Model to be trained
        pth_to_state_dict (str, optional): Load previous checkpoint if needed. Defaults to None.
        device (torch.device, optional): Device to store model. Defaults to torch.device('cpu') (CUDA not supported)

    TODO:
        Support cuda in future
    Returns:
        [type]: [description]
    """
    global g_net, g_lock, g_version


    # Ensure that we get only one handle to the net.
    if g_net is None:
        # construct it once
        g_net = model(*args, **kwargs)
        if pth_to_state_dict is not None:
            g_net.load_state_dict(torch.load(pth_to_state_dict))
        g_net.to(device)
        g_net.train()
        g_version = 0

    return g_net

def _init_optimizer(optim_class: Optimizer, lr: float=1e-6, *args, **kwargs):
    """Init optimizer
    Remark:
        Call this method after _init_net() or you will get error
    """
    global g_optimizer
    if g_optimizer is None:
        g_optimizer = optim_class(g_net.parameters(), lr=lr, *args, **kwargs)
    return g_optimizer



def init(policy: Dict[str, Any]) -> None:
    global g_policy
    g_policy = policy

    _init_share()
    _init_net(Net)
    _init_optimizer(torch.optim.SGD, policy['lr'])
    pass


if __name__ == '__main__':
    DEGUG: bool = True

    parser = argparse.ArgumentParser(description="Parameter-Server HTTP based training")

    parser.add_argument("--master_addr",
                        type=str,
                        default="0.0.0.0",
                        help="""Address of master, will default to 0.0.0.0 if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument("--master_port",
                        type=str,
                        default="29500",
                        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument("--batch_sz",
                        type=int,
                        default=4,
                        help="""Batch size of FedSGD""")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="""Batch size of FedSGD""")

    args = parser.parse_args()
    master_addr: str = args.master_addr
    master_port: str = args.master_port

    print(f'[ Info ] Start server at {master_addr}:{master_port}')

    init({"batch_sz":args.batch_sz, "lr":args.lr})
    # Warning: This app does not support multi-process server yet
    # Warning: On Mac OS X, Queue.qsize() is not implemented
    if DEGUG:
        app.run(master_addr, master_port)
    else:
        server = pywsgi.WSGIServer((master_addr, master_port), app)
        server.serve_forever()

```

## Design Docker Contained Worker

The worker will be launched by Openwhisk. Therefore, two api `/run` and `/init` are needed.

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
        "dataset_url": "http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7",
        "device": "cpu"
    }
}
```

Where the key `value` is automatically added by OpenWhisk.

Full code of worker:

```python
"""run_worker.py"""
from typing import Any
import urllib
import os
import time
import tarfile
import json

import torch
import torch.nn as nn

import tqdm
from flask import Flask, request, Response
from gevent import pywsgi

from models import Net
from worker_utils import *

g_net: nn.Module = None
g_train_dataset: torch.utils.data.Dataset
app = Flask(__name__)

g_train_dataset = None

def import_train_dataset_from_url(url: str) -> bool:
    global g_train_dataset
    print(f"[ Info ] Downloading dataset from {url}")
    try:
        urllib.request.urlretrieve(url, 'tmp.tar.gz')
    except Exception:
        if os.path.exists('./tmp.tar.gz'):
            os.remove('./tmp.tar.gz')
        raise Exception
    
    tar_file = tarfile.open('./tmp.tar.gz')
    tar_file.extractall()
    
    num_retry: int=5
    while num_retry > 0:
        try:
            import dataset_dl
            break
        except:
            time.sleep(1)
            num_retry -= 1
            pass

    g_train_dataset = dataset_dl.Dataset
    print(f"[ Info ] Successfully imported dataset: {g_train_dataset.__repr__()}")

    return True


@app.route("/run", methods=['GET', 'POST'])
def run():
    resp = Response(mimetype='application/json')
    try:
        policy = request.get_json()["value"] # extract real arguments
        print(policy)
    except KeyError:
        print("[ Error ] No argument")
        resp.status = 500
        resp.data = json.dumps({"code": 500, "msg": "No argument"})
        return resp
    
    try:
        BATCH_SZ_TRAIN: int = policy["batch_sz_train"]
        EPOCH_N: int = policy["epoch_n"]
        APIHOST: str =  policy["apihost"]
        UPDATE_INTV: int =  policy["update_intv"]
        DATASET_URL: Any =  policy["dataset_url"]
        DEVICE = torch.device( policy["device"])
    except KeyError:
        print("[ Error ] Wrong parameters")
        resp = Response(status=500)
        resp.data = json.dumps({"code": 500, "msg": "Wrong parameters"})
        return resp

    global g_net, g_train_dataset

    if g_net is None: init()

    assert import_train_dataset_from_url(DATASET_URL)

    g_net.to(DEVICE)
    state_dict = get_param_from_remote(APIHOST)
    g_net.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(g_train_dataset,
                                               batch_size=BATCH_SZ_TRAIN,
                                               shuffle=True)

    # Model version
    local_version: int = get_version_from_remote(APIHOST)

    for epoch_idx in range(1, EPOCH_N + 1):
        train_loss_tot: float = 0.0
        train_loss_cnt: int = 0
        with tqdm.tqdm(range(len(train_loader))) as pbar:
            for batch_idx, (stimulis, label) in enumerate(train_loader):
                pred = g_net(stimulis.to(DEVICE))
                # label = torch.nn.functional.one_hot(label, num_classes=10).to(pred.dtype)
                loss = torch.nn.functional.cross_entropy(pred, label.to(DEVICE))
                loss.backward()
                
                train_loss_tot += float(loss.detach().cpu().numpy())
                train_loss_cnt += 1

                if batch_idx % UPDATE_INTV == 0:
                    local_grad = get_grad_from_local(g_net, 1/UPDATE_INTV)
                    ret = put_grad_to_remote(APIHOST, local_grad)
                
                remote_version = get_version_from_remote(APIHOST)
                if remote_version > local_version:
                    local_version = remote_version
                    state_dict = get_param_from_remote(APIHOST)
                    g_net.load_state_dict(state_dict)
                    
                pbar.set_description(f"version: {local_version}loop: {epoch_idx}, avg_loss:{train_loss_tot / train_loss_cnt}")
                pbar.update(1)
    resp.status = 500
    resp.data = json.dumps({"code": 200, "avg_loss": train_loss_tot / train_loss_cnt})
    return resp

@app.route("/init", methods=['GET', 'POST'])
def init():
    global g_net
    g_net = Net()
    g_net.train()


if __name__ == '__main__':
    DEGUG: bool = True
    SERVING_PORT: int = 8080
    
    if DEGUG:
        app.run('0.0.0.0', SERVING_PORT)
    else:
        server = pywsgi.WSGIServer(('0.0.0.0', SERVING_PORT), app)
        server.serve_forever()
```

```python
"""worker_utils.py"""
import requests
import pickle
import torch
import torch.nn as nn
from typing import List, Dict, Union, Any


def get_param_from_remote(apihost: str) -> Union[None, Dict[str, torch.Tensor]]:
    """Get state dict(parameters) from a remote api

    Args:
        apihost (str): url for api

    Returns:
        Union[None, Dict[str, torch.Tensor]]: If succeed, a dictionary of parameters will be obtained
    """
    apihost += '/getParameter'
    model_param = None
    resp = requests.get(apihost)
    if 'application/octet-stream' in resp.headers['Content-Type']:
        resp_dict: Dict[str, Any] = pickle.loads(resp.content)
    else:
        return None

    assert isinstance(resp_dict, dict)
    assert "code" in resp_dict.keys()
    if resp_dict["code"] == 200:
        model_param = resp_dict["param"]

    return model_param


def get_version_from_remote(apihost: str) -> int:
    """Get model parameter version from remote

    Args:
        apihost (str): url for api

    Returns:
        int: version of model parameter
    """
    apihost += '/getVersion'
    resp = requests.get(apihost)
    if resp.status_code != 200:
        return -1
    else:
        resp_dict = resp.json()
        if 'code' in resp_dict and resp_dict['code'] == 200:
            return resp_dict['version']
        else:
            return -1


def get_grad_from_local(net: nn.Module, gain: float = 1.0) -> Dict[str, torch.Tensor]:
    """Bundle parameters of a local nn.Module to Dict

    Args:
        net (nn.Module): [description]

    Returns:
        Dict[str, torch.tensor]: parameter list
    """

    gradient_dict = {}
    module_parameters: List[str, nn.parameter.Parameter] = list(net._named_members(lambda module: module._parameters.items()))
    for name, param in module_parameters:
        gradient_dict[name] = (param.grad.clone().detach() * gain).cpu()
    return gradient_dict


def put_grad_to_remote(apihost: str, grad_dict: Dict[str, torch.Tensor], worker_id: Any = 0):
    """Put gradient to remote parameter server

    Args:
        apihost (str): URL of parameter server api
        grad_dict (Dict[str, torch.Tensor]): dict to put
        worker_id (Any, optional): The id of current worker. Defaults to 0.

    Returns:
        bool: [description]
    """
    apihost += '/putGradient'
    req = requests.post(url=apihost,
                        data=pickle.dumps({
                            "id": worker_id,
                            "param": grad_dict
                        }),
                        headers={"Content-Type": 'application/octet-stream'})
    if req.status_code != 200:
        return False

    if 'code' in req.json().keys() and req.json()['code'] == 200:
        return True
    else:
        return False
```

## Test distributed training

First, we split mnist dataset into 6 parts. A `dataset_dl` module is created :

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

`__init__.py`

```python
"""__init__.py"""
from .__dataset__ import dataset as Dataset
```

`__dataset__.py`

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
> dataset = dataset_dl.Dataset
> ```
>
> without knowiing the detail of implementation

We then split the dataset by 6

```bash
tar -zcvf dataset_dl_1.tar.gz ./dataset_dl # Subset(0,10000)
tar -zcvf dataset_dl_2.tar.gz ./dataset_dl # Subset(10000,20000)
tar -zcvf dataset_dl_3.tar.gz ./dataset_dl # Subset(20000,30000)
tar -zcvf dataset_dl_4.tar.gz ./dataset_dl # Subset(30000,40000)
tar -zcvf dataset_dl_5.tar.gz ./dataset_dl # Subset(40000,50000)
tar -zcvf dataset_dl_6.tar.gz ./dataset_dl # Subset(50000,60000)
```

We upload these datasets to OSS with the help of previously created`file_uploader.py`:

```bash
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_1.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'd96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/d96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090622Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e44304c335bbd5ef00c8726eef207d8b3448956e35b072d60a5f4d4af08b0987'}
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_2.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'dd3a1f9c-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/dd3a1f9c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090628Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a1ba60ed965430d62752a2f58431affea5ebd63e86ee78aaaeb4593a00a3b03'}
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_3.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'e0d827fc-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/e0d827fc-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090634Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=3753bd8d0b0f17068a4d870b66971e0749ee47a235f3f35eca42f4e15cb2726e'}
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_4.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'e4585d48-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/e4585d48-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090640Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9e5e632d180751923e46b3aaac41904fc643e4eba3bcc85a58c69fc6ba7eda11'}
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_5.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'e79e3e3c-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/e79e3e3c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090646Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=4408d50e5909013c0673d2829d2c776450f20abcb78aa73227fa41d427ef01e6'}
$ python file_uploader.py --endpoint=192.168.1.131:9000 --access_key=testAccessKey --secret_key=testSecretKey --bucket_name=mnist-dataset --file=dataset_dl_6.tar.gz
{'bucket_name': 'mnist-dataset', 'object_name': 'ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz', 'url': 'http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7'}
```

### Summary of urls

| Volume | URL                                                                                                                                                                                                                                                                                                                                                   |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | [http://192.168.1.131:9000/mnist-dataset/d96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090622Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e44304c335bbd5ef00c8726eef207d8b3448956e35b072d60a5f4d4af08b0987](http://192.168.1.131:9000/mnist-dataset/d96e3d6c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090622Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e44304c335bbd5ef00c8726eef207d8b3448956e35b072d60a5f4d4af08b0987) |
| 2      | [http://192.168.1.131:9000/mnist-dataset/dd3a1f9c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090628Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a1ba60ed965430d62752a2f58431affea5ebd63e86ee78aaaeb4593a00a3b03](http://192.168.1.131:9000/mnist-dataset/dd3a1f9c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090628Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a1ba60ed965430d62752a2f58431affea5ebd63e86ee78aaaeb4593a00a3b03) |
| 3      | [http://192.168.1.131:9000/mnist-dataset/e0d827fc-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090634Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=3753bd8d0b0f17068a4d870b66971e0749ee47a235f3f35eca42f4e15cb2726e](http://192.168.1.131:9000/mnist-dataset/e0d827fc-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090634Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=3753bd8d0b0f17068a4d870b66971e0749ee47a235f3f35eca42f4e15cb2726e) |
| 4      | [http://192.168.1.131:9000/mnist-dataset/e4585d48-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090640Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9e5e632d180751923e46b3aaac41904fc643e4eba3bcc85a58c69fc6ba7eda11](http://192.168.1.131:9000/mnist-dataset/e4585d48-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090640Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9e5e632d180751923e46b3aaac41904fc643e4eba3bcc85a58c69fc6ba7eda11) |
| 5      | [http://192.168.1.131:9000/mnist-dataset/e79e3e3c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090646Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=4408d50e5909013c0673d2829d2c776450f20abcb78aa73227fa41d427ef01e6](http://192.168.1.131:9000/mnist-dataset/e79e3e3c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090646Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=4408d50e5909013c0673d2829d2c776450f20abcb78aa73227fa41d427ef01e6) |
| 6      | [http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7](http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7) |

> The dataset on OSS (data and `.py` descriptors) must be tar.gz files

### API test with curl

```bash
curl -X POST \
     -d '{"value":{"batch_sz_train": 32, "epoch_n": 32, "apihost": "http://192.168.1.207:29500","update_intv": 8, "dataset_url": "http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7","device": "cpu"}}' \
     -H 'Content-Type: application/json' http://localhost:8080/run
```

After local test is completed. We construct the docker image that runs the worker

```bash
docker build . -t python3action-dist-train-mnist
```

Test the containner locally with curl

```bash
docker run --rm --net=host python3action-dist-train-mnist
curl ...
```

Tag the image and upload

```bash
docker login
docker tag python3action-dist-train-mnist natrium233/python3action-dist-train-mnist:1.0
docker push natrium233/python3action-dist-train-mnist:1.0  
```

## Configure OpenWhisk

Create OpenWhisk action

```bash
wsk action create dist-train --docker natrium233/python3action-dist-train-mnist:1.0 --web true --timeout 300000
```

Invoke action

```bash
wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.1.207:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.1.131:9000/mnist-dataset/ea8105d0-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090651Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=9a7738e744b88077e0132946b82bc158e283a1ab0e0a5ffc9f295d3a84d357d7' \
    --param device cpu
```

```bash
wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.1.207:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.1.131:9000/mnist-dataset/dd3a1f9c-1ddf-11ec-9aeb-c3cd4bc871fd.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=testAccessKey%2F20210925%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210925T090628Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a1ba60ed965430d62752a2f58431affea5ebd63e86ee78aaaeb4593a00a3b03' \
    --param device cpu
```

> Parameter server is at `192.168.1.207`
> OSS server is at `192.168.1.131`

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

The 4GB virtual machine I created have 2.5GiB of free memory at idle stat. But since each activation consumes 33% of CPU, the virtual machine can only hold up to 3 workers at one time

If we insist to add workers on this machine, the trainning will be bottenecked by CPU

If we could deploy OpenWhisk to a Kubernetes cluster, the number of workers can increase. What is more, if the cluster had GPU installed and `nvidia-docker` installed, the traininig would be accelerated by GPU.

However, configuring `nvidia-docker` on a Kubernetes cluster is tredious.

Pay attention to OpenWhisk timeout policy. On a standalone server, an action must finish within its timeout (which is 300000 milliseconds maximum).
