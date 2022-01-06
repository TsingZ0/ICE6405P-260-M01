import torch
from typing import Callable, Dict, Union, List
from io import BytesIO
from PIL import Image
import minio
from torchvision import transforms
from model import LeNet5
import sys

net: torch.nn.Module = None

DEFAULT_TRANSFORM: Callable = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize(28),
     transforms.ToTensor()])

MINIO_CREDENTIAL: Dict[str, str] = {
    "endpoint": "192.168.1.82:9000",
    "access_key": "testAccessKey",
    "secret_key": "testSecretKey"
}

PATH_TO_STATE_DICT: str = "/opt/mnist/model.pth"

@torch.no_grad()
def decode_picture_from_oss(
        obj_info: Dict[str, str],
        credential: Dict[str, str]) -> Union[None, torch.Tensor]:
    """Decode a picture from OSS service
    http://{$ENDPOINT}/api/v1/buckets/{$BUCKET_NAME}/objects/download?prefix={$OBJECT_NAME}&access_key={$ACCESS_KEY}&secret_key={$SECRET_KEY}

    Args:
        obj (Dict[str, str]): 
        {
            "bucket_name": "mnist",
            "object_name":"082d97b2-19f1-11ec-a558-1e00d10c4441.png"
        }

        credential (Dict[str, str]): 
        {
            "endpoint": "192.168.1.82",
            "access_key": "testAccessKey",
            "secret_key": "testSecretKey"
        }

    Returns:
        torch.Tensor: Tensor of shape (1,1,28,28)
    """
    # base_url = "http://{endpoint}/api/v1/buckets/{bucket_name}/objects/download?prefix={object_name}&access_key={access_key}&secret_key={secret_key}"
    # url = base_url.format(**(obj | credential))
    # print(url)

    try:
        client = minio.Minio(**credential, secure=False)
        obj_info = client.get_object(**obj_info)
    except Exception as err:
        print(err)
        return None

    try:
        img = Image.open(BytesIO(obj_info.read()))
        img = DEFAULT_TRANSFORM(img)
        return img.unsqueeze(0)
    except Exception as err:
        print(err)
        return None


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
    sys.path.append('/opt/mnist')

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


if __name__ == '__main__':
    import json
    object_infos = json.loads(sys.argv[1])
    credential = MINIO_CREDENTIAL
    print(main({'credential': credential, 'object_infos': object_infos}))
