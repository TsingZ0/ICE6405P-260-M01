import argparse
import os
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torchvision import datasets, transforms

import tqdm
from gevent import pywsgi
from flask import Flask, jsonify, request, Response
from typing import Dict, Iterable, List, Any

from utils import remote_method
from models import Net

app = Flask(__name__)

# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, model: nn.Module, num_gpus=0):
        super().__init__()
        net = model(num_gpus=num_gpus)
        self.net = net
        self.input_device = torch.device("cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.net(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.net.parameters()]
        return param_rrefs

param_server = None
global_lock = Lock()

def get_parameter_server(model: nn.Module, num_gpus=0):
    global param_server
    global global_lock
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if param_server is None:
            # construct it once
            param_server = ParameterServer(model, num_gpus=num_gpus)
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    global param_server

    print("[ Info ] PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print(f"[ Info ] RPC initialized! Running parameter server, rank={rank}, world_size={world_size}")
    rpc.shutdown()

    # TODO: Save model after training
    print("[ Info ] RPC shutdown on parameter server.")

# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, model: nn.Module, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote("parameter_server", get_parameter_server, args=(model, num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(ParameterServer.get_param_rrefs, self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(ParameterServer.forward, self.param_server_rref, x)
        return model_output


def run_training_loop(model: nn.Module, rank, num_gpus: int, train_loader: Iterable, test_loader: Iterable):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    print("[ Info ] Run training loop")
    trainer_net = TrainerNet(model, num_gpus=num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = trainer_net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
    with tqdm.tqdm(range(len(train_loader))) as pbar:
        for i, (data, target) in enumerate(train_loader):
            with dist_autograd.context() as cid:
                model_output = trainer_net(data)
                target = target.to(model_output.device)
                loss = F.nll_loss(model_output, target)
                if i % 5 == 0:
                    pbar.set_description(f"[ Info ] Rank {rank} training batch {i} loss {loss.item()}")
                dist_autograd.backward(cid, [loss])
                # Ensure that dist autograd ran successfully and gradients were
                # returned.
                assert remote_method(ParameterServer.get_dist_gradients, trainer_net.param_server_rref, cid) != {}
                opt.step(cid)
                pbar.update()


    print("[ Info ] Training complete!")
    print("[ Info ] Getting accuracy....")

    # Do some test and return result
    accuracy = get_accuracy(test_loader, trainer_net)
    return {"accuracy": accuracy}


def get_accuracy(test_loader, model) -> float:
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0 and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"[ Info ] Accuracy {correct_sum / len(test_loader.dataset)}")
    return correct_sum / len(test_loader.dataset)


# Main loop for trainers.
def run_worker(model: nn.Module, rank, world_size, num_gpus, train_loader, test_loader):
    print(f"[ Info ] Worker rank {rank} initializing RPC")
    try:
        ret = rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=world_size)
    except RuntimeError as err:
        if str(err) == 'RPC is already initialized':
            print('[ Info ] RPC already inited')
        else:
            raise err


    print(f"[ Info ] Worker {rank} done initializing RPC")

    ret = run_training_loop(model, rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()
    return ret

@app.route("/run", methods=['POST', 'GET'])
def run():
    """Train the model in this method
    1. Receive json data from POST request: 
        {
            "world_size":n,
            "rank":1,
            "num_gpus":0,
            "master_addr":"localhost"
            "master_port":29500
        } TODO: Add key train_data to request data
    2. Retrieve distributed training data
    3. Execute run_worker

    Returns:
        json: Json formatted string
    """

    try:
        # extract real arguments
        worker_info: Dict[str, Any] = request.get_json()["value"]
        print(worker_info)
    except KeyError:
        print("[ Error ] No argument")
        return jsonify({"code": 500, "res": {}})

    # Check keys
    assert 'rank' in worker_info.keys()
    assert 'world_size' in worker_info.keys()
    assert 'num_gpus' in worker_info.keys()
    assert 'master_addr' in worker_info.keys()
    assert 'master_port' in worker_info.keys()

    rank = worker_info['rank']
    world_size = worker_info['world_size']
    num_gpus = worker_info['num_gpus']
    master_addr = worker_info['master_addr']
    master_port = worker_info['master_port']
    
    # Master address and master port are set via environment variables
    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(master_port)
    print(f'[ Info ] Set master server as {master_addr}:{master_port}')

    # Get data to train on
    # TODO: Train data should get from OSS, with parameters in json
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307, ), (0.3081, ))])),
                                               batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        './data',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307, ), (0.3081, ))])),
                                              batch_size=32,
                                              shuffle=True)

    ret = run_worker(Net, rank, world_size, num_gpus, train_loader, test_loader)

    # TODO cleaning, The code runs in a container. The container might be reused
    return jsonify({"code": 200, "res": ret})
    


@app.route("/init", methods=['POST', 'GET'])  # app.route does not accept POST actions by default
def init():
    """There is actually nothing to init, due to the ambigious of num_gpu, world_size and rank

    Returns:
        [type]: [description]
    """
    return Response("OK", status=200, mimetype='text/html')


# --------- Launcher --------------------

if __name__ == '__main__':
    DEGUG: bool = True
    SERVING_PORT: int = 8080

    parser = argparse.ArgumentParser(description="Parameter-Server RPC based training")
    parser.add_argument("--mode", type=str, default='worker', help="Type of node, could be server or worker")
    parser.add_argument("--world_size",
                        type=int,
                        default=4,
                        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=0,
                        help="""Number of GPUs to use for training, currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument("--master_addr",
                        type=str,
                        default="localhost",
                        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument("--master_port",
                        type=str,
                        default="29500",
                        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    if args.mode == 'server':
        print("[ Info ] Runnning in server mode")
        rank = 0 # Server rank defaults to 0
        assert args.num_gpus <= 3, f"[ Error ] Only 0-2 GPUs currently supported (got {args.num_gpus})."
        
        # Master address and master port are set via environment variables
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        print(f'[ Info ] Start server at {args.master_addr}:{args.master_port}')

        processes = []
        world_size = args.world_size
        run_parameter_server(0, world_size)

        # p = mp.Process(target=run_parameter_server, args=(0, world_size))
        # p.start()
        # processes.append(p)
        # try:
        #     for p in processes:
        #         p.join()
        # except KeyboardInterrupt:
        #     for p in processes:
        #         p.terminate()

    elif args.mode == 'worker':
        print("[ Info ] Runnning in worker mode")

        if DEGUG:
            app.run('0.0.0.0',SERVING_PORT)
        else:
            server = pywsgi.WSGIServer(('0.0.0.0', SERVING_PORT), app)
            server.serve_forever()
    else:
        raise ValueError(f"[ Error ] Working mode {args.mode} not legal")