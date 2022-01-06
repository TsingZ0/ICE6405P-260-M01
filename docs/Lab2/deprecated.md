# Deprecated

## Wrap the model

We wrap the model using a `main()` function in `deploy-docker.py`:

```python
def main(params) -> List[Dict]:
    """infer an hand-written digit
    1. Receive list of json formatted POST request: 
        [{
            "bucket_name":"mnist,
            "object_name":"082d97b2-19f1-11ec-a558-1e00d10c4441.png"
        },...]
    2. Get the Image from OSS
    3. Infer Image
    4. Return the result

    Returns:
        [type]: [description]
    """
    global net

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    net = LeNet5()
    net.load_state_dict(torch.load(PATH_TO_STATE_DICT))
    net.to(device)
    net.eval()

    object_infos: List[str] = params['object_infos']
    credential = params['credential']

    length = len(object_infos)
    res: List[Dict[str, int]] = []

    for obj_info in object_infos:
        stimulis = decode_picture_from_oss(obj_info, credential)
        if stimulis is not None:
            pred = net(stimulis)
            pred_decoded = torch.argmax(pred, dim=1)
            pred_as_int = int(pred_decoded.cpu().numpy())
            res.append({"code": 200, "pred": pred_as_int} | obj_info)
        else:
            res.append({"code": 500, "pred": -1} | obj_info)

    return json.dumps({"result": res})
```

We tested the wrapped function and it passed the test:

```python
if __name__ == '__main__':
    import sys, json
    object_infos = json.loads(sys.argv[1])
    credential = MINIO_CREDENTIAL
    print(main({'credential': credential, 'object_infos': object_infos}))
```

```bash
$ python deploy-docker.py "[{\"bucket_name\":\"mnist\", \"object_name\":\"082d97b2-19f1-11ec-a558-1e00d10c4441.png\"}]"
{"result": [{"code": 200, "pred": 6, "bucket_name": "mnist", "object_name": "082d97b2-19f1-11ec-a558-1e00d10c4441.png"}]}
```

### DPDK

Install `meson`

```bash
sudo apt-get install meson
```

Install dpdk

```bash
wget https://fast.dpdk.org/rel/dpdk-20.11.1.tar.xz && tar xf dpdk-20.11.1.tar.xz
export DPDK_DIR=$(pwd)/dpdk-stable-20.11.1
cd $DPDK_DIR
export DPDK_BUILD=$DPDK_DIR/build
meson build
ninja -C build
sudo ninja -C build install
sudo ldconfig
export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH
```

Check version

```bash
pkg-config --modversion libdpdk
```

```bash
sudo apt-get install libpcap-dev libnuma-dev
wget http://fast.dpdk.org/rel/dpdk-19.11.tar.xz
tar xf dpdk-19.11.tar.xz
make config T=x86_64-native-linuxapp-gcc

### OpenVSwitch

```bash
git clone https://github.com/openvswitch/ovs.git
git checkout v2.7.0
./boot.sh
./configure --with-dpdk=yes
make
```
