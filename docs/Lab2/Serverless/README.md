# Serverless

- [Serverless](#serverless)
  - [Introduction](#introduction)
  - [Getting OpenWhisk ready](#getting-openwhisk-ready)
    - [Install Docker](#install-docker)
      - [Install JAVA](#install-java)
      - [Install NodeJS, NPM](#install-nodejs-npm)
    - [Get OpenWhisk source code and Configure](#get-openwhisk-source-code-and-configure)
      - [Trouble shooting: Cannot connect to docker daemon due to privillege](#trouble-shooting-cannot-connect-to-docker-daemon-due-to-privillege)
    - [Get OpenWhisk CLI](#get-openwhisk-cli)
    - [Launch OpenWhisk](#launch-openwhisk)
    - [Verify OpenWhisk Installation](#verify-openwhisk-installation)
  - [Deploy MinIO, an Open-source Object Storage Service](#deploy-minio-an-open-source-object-storage-service)
  - [Deploy ML Applications](#deploy-ml-applications)
    - [Train the Model](#train-the-model)
    - [Convert the model to ONNX](#convert-the-model-to-onnx)
    - [Wrap the ML model as a WebApp](#wrap-the-ml-model-as-a-webapp)
    - [The app](#the-app)
    - [Create Docker Image for this ML application](#create-docker-image-for-this-ml-application)
  - [Configure OpenWhisk docker action](#configure-openwhisk-docker-action)
    - [Creating OpenWhisk API](#creating-openwhisk-api)
  - [Summary](#summary)
  - [Appendix - Convenient setup script](#appendix---convenient-setup-script)

## Introduction

The goal of this project is to deploy a ML inference app on Openwhisk. The app can be viewed as a function

$$
\mathtt{Result} = Infer(\mathtt{image})
$$

Under the Openwhisk limit, the image should be passed to funtion in URL.

The experiment setting is as follows

- Environment: Ubuntu 20.04LTS
- Platform: Windows Hyper-V, x86_64

> The IP address of target VM is `192.168.1.82`

## Getting OpenWhisk ready

In this section, we deploy a **standalone** openwhisk stack on target machine.

### Install Docker

We manually install docker using convenient script from [get-docker.com](get-docker.com)

```bash
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh
 ```

To accelerate download, we add our own registory to `/etc/docker/daemon.json`

```bash
$ sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://p3e80qty.mirror.aliyuncs.com"]
}
EOF
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

#### Install JAVA

```bash
$ sudo apt-get install openjdk-11-jdk
```

#### Install NodeJS, NPM

```bash
$ sudo apt-get install nodejs
$ sudo apt-get install npm
```

### Get OpenWhisk source code and Configure

You will find OpenWhisk's source code on Github. We clone the code and build it with gradlew.

```bash
$ git clone --recursive https://github.com/apache/openwhisk.git
$ cd openwhisk
$ ./gradlew core:standalone:build
```

#### Trouble shooting: Cannot connect to docker daemon due to privillege

```bash
$ sudo usermod -aG docker $USER 
$ newgrp docker
```

If the `docker` group does not exist:

```bash
$ sudo groupadd docker
```

Restart the docker service

```bash
$ sudo systemctl restart docker
```

Verify normal user can use docker

```bash
$ docker run hello-world
```

Stop the gradle daemon, then try again to deploy

```bash
$ ./gradlew --stop
$ ./gradlew core:standalone:build
```

### Get OpenWhisk CLI

To connect and manipulate OpenWhisk, `wsk` CLI is needed. We install `wsk` from [openwhisk-cli](https://github.com/apache/openwhisk-cli) repository:

```bash
$ wget https://github.com/apache/openwhisk-cli/releases/download/1.2.0/OpenWhisk_CLI-1.2.0-linux-amd64.tgz
$ tar -xzvf OpenWhisk_CLI-1.2.0-linux-amd64.tgz 
$ mv wsk /usr/local/bin/
```

> The `amd64` postfix should be replaced with proper platfrom

### Launch OpenWhisk

After the build process has finished, launch openwhisk using `java -jar`:

```bash
$ java -jar ./bin/openwhisk-standalone.jar
```

> This is a foreground task, do not close the terminal or the OpenWhisk will stop

![OpenWhisk](img/1.png)

Credential of our OpenWhisk instance is printed to `stdout` (as highlighted in the picture above). Use this credential to setup `wsk` cli.

```bash
$ wsk property set --apihost 'http://172.17.0.1:3233' \
                   --auth ...
```

Now we can manipulate OpenWhisk with `wsk` command.

### Verify OpenWhisk Installation

We create a `hello.js` file to test if our OpenWhisk installation is operational

```bash
$ echo "function main(params){" \
        "var name = params.name || 'World';" \
        "return {payload:  'Hello, ' + name + '\!'};}" > hello.js
```

We then create an OpenWhisk Action that binds to this script:

```bash
$ wsk action create helloworld hello.js
ok: created action helloworld
```

To invoke this action, simply run:

```bash
$  wsk action invoke helloworld --result --param name david
{
    "playload": "Hello, david!"
}
```

## Deploy MinIO, an Open-source Object Storage Service

To store pictures of handwritten digits, we need some sort of storage services. The `MinIO` server is a decent choice. It's an object storage that support url sharing.

```bash
$ docker run \
    --name minio \
    --net=host \
    -d \
    -v $PATH_TO_DATA:/data \
    -v $PATH_TO_CONFIG:/root/.minio \
    -e "MINIO_ROOT_USER=$CUSTOM_ROOT_USER" \
    -e "MINIO_ROOT_PASSWORD=$CUSTOM_ROOT_PASSWORD" \
    minio/minio \
    server /data --console-address $ADDRESS
```

For example:

```bash
$ docker run \
    --name minio \
    --net=host \
    -d \
    -v /home/speit/minio-data:/data \
    -v /home/speit/minio-config:/root/.minio \
    -e "MINIO_ROOT_USER=root" \
    -e "MINIO_ROOT_PASSWORD=password" \
    minio/minio \
    server /data --console-address "0.0.0.0:41309"
```

Using the WebUI of MinIO, we create :

1. A bucket named `minist`
2. A new credential `testAccessKey:testSecretKey`

![MinIO](img/2.png)

An upload function is created to upload photos to this OSS service:

```python
"""file_uploader.py
"""
from minio import Minio
import argparse
import uuid
from typing import Union, Dict
import os
from datetime import timedelta


def oss_upload(endpoint: str,
               bucket_name: str,
               access_key: str,
               secret_key: str,
               file: str,
               object_name: str = '') -> Union[None, Dict[str, str]]:
    """Upload a file to MinIO OSS

    Args:
        args (ArgumentParser): should contain following attributes:
            - args.endpoint: str, url
            - args.access_key: str
            - args.secret_key: str

    Returns:
        Dict[str, str]: {$BUCKET:$NAME}
    """
    minioClient = Minio(endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=False)  # Create MinIO client
    try:
        if len(object_name) <= 0:
            object_name = str(uuid.uuid1()) + os.path.splitext(file)[
                -1]  # Generate unique object name
        else:
            object_name = object_name

        minioClient.fput_object(bucket_name, object_name,
                                file)  # Upload object
        url = minioClient.presigned_get_object(bucket_name,
                                               object_name,
                                               expires=timedelta(days=2))
        ret = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "url": url
        }  # Return the object info
        return ret

    except Exception as err:
        print(err)
        return None


if __name__ == '__main__':
    """Usage

    python file_upload.py --endpoint=192.168.1.82:9000 \
                          --access_key=testAccessKey \
                          --secret_key=testSecretKey \
                          --bucket=mnist \
                          --file=mnist.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str)
    parser.add_argument('--access_key', type=str)
    parser.add_argument('--secret_key', type=str)
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--object_name', type=str, default='')
    args = parser.parse_args()
    print(
        oss_upload(endpoint=args.endpoint,
                   bucket_name=args.bucket_name,
                   access_key=args.access_key,
                   secret_key=args.secret_key,
                   file=args.file,
                   object_name=args.object_name))
```

It returns the name of bucket and the name of file as a dictionary, together with an encrypted url:

```json
{
    "bucket_name": "mnist",
    "object_name":"082d97b2-19f1-11ec-a558-1e00d10c4441.png",
    "url":"http://192.168.1.82:9000/mnist/082d97b2-19f1-11ec-a558-1e00d10c4441.png?..."
}
```

## Deploy ML Applications

### Train the Model

We follow [this paper](https://ieeexplore.ieee.org/document/726791?arnumber=726791) to build a multi-layer CNN model using Pytorch. The model is trained with MNIST dataset.

### Convert the model to ONNX

With the help of `torch.onnx.export`, we convert our trained model to ONNX format for deployement.

### Wrap the ML model as a WebApp

With the help of [Flask](https://flask.net.cn), the ML model can be wrapped as a WebApp.

> See `deploy-flask.py` for details

The WebApp we designed can be accessed by HTTP GET/POST actions. For example:

```console
$ curl -X POST -d '{"value":{"url":"http://192.168.1.82:9000/mnist/test_picture.png"}}' \
       -H 'Content-Type: application/json' http://localhost:8080/run
{"code":200,"res":6}
```

### The app

1. Init the model
2. Get image from url
3. Generate prediction

These information can be obtained from `upload()` function.

If the requested object does not exist, or is not readable, the service returns `{"code":500, "res":-1}`

### Create Docker Image for this ML application

OpenWhisk support Docker actions, but has several limits. According to [OpenWhisk Docker Actions](https://jamesthom.as/2017/01/openwhisk-docker-actions/):

- The action should offer a RESTful API at port `8080`
- The action should offer two api

  1. HTTP POST at `/init`
  2. HTTP POST at `/run` with JsonData

`/init` API will be called to set envrionment variables. It is automatically called by OpenWhisk in the prewarm stage. After the prewarm, each invoke of action will simple pass parameters to `/run` in a specific JSON format:

```json
{
    "value":{ // OpenWhisk default
        "param1":"value1",
        "param2":"value2"
    }
}
```

We come up with the following Dockerfile to build our custom ML image:

- The image is based on python:3.8.8-slim to reduce size
- It installs flask, pytorch and other neccessary packages
- It launches our flask ML application

```Dockerfile
FROM python:3.8.8-slim

ENV LANG=C.UTF-8

WORKDIR /opt/app/

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install minio pillow tqdm flask gevent requests onnx onnxruntime

ENV PATH_TO_ONNX_MODEL=/opt/app/model.onnx

COPY ["model.onnx", "deploy-flask.py", "/opt/app/"]

EXPOSE 8080

CMD ["/bin/bash", "-c", "python /opt/app/deploy-flask.py"]

```

> Known Issus:
>
> opehwhisk/python3action-base is based on `alphine`, which is not shipped with `glibc` but with `muslc`, an replacement of standard C library. Therefore we cannot easily run `miniconda` (which requires `glibc`) on it.

We build the image using this command:

```bash
$ docker build . -t python3action-mnist
```

Test run:

```bash
$ docker run -it --rm --net=host python3action-mnist
```

OpenWhisk docker actions requires docker image to be public, therefore we need to push our image to docker hub:

```bash
$ docker tag python3action-mnist $USER_NAME/python3action-mnist:1.0
$ docker login
$ docker push $USER_NAME/python3action-mnist:1.0
```

> It is extremely dangerous to store credentials in python script. Consider using environment variables to store access_key and secret_key
>
> This image runs on `x86_64`. **NOT** `arm`.
>
> For docker cross-platform build, visit [this site](https://printempw.github.io/build-x86-docker-images-on-an-m1-macs/)
>
> ```bash
> docker buildx create --use --name m1_builder
> docker buildx build . --platform linux/amd64 -t python3action-mnist --load
> ```

Test API

```bash
$ curl http://localhost:8080/init
OK
```

```bash
$ curl -X POST -d '{"value":{"url": "http://192.168.1.82:9000/mnist/test_picture.png"}}' \
       -H 'Content-Type: application/json' http://localhost:8080/run
{"code":200,"res":6}
```

## Configure OpenWhisk docker action

Create action by:

```bash
$ wsk action create mnist --docker natrium233/python3action-mnist:1.0
```

If the action is created, use the following command to update

```bash
$ wsk action update mnist --docker natrium233/python3action-mnist:1.2
```

Invoke action with following command

```bash
$ wsk action invoke mnist --result --param url "http://192.168.1.82:9000/mnist/test_picture.png"
{
    "code": 200,
    "res": 6
}
```

### Creating OpenWhisk API

> Reference [adobeio-runtime](https://adobedocs.github.io/adobeio-runtime/guides/creating_rest_apis.html)

To use OpenWhisk via RESTful api, we have to launch our OpenWhisk stack with `--api-gw` enabled:

```bash
java -jar openwhisk/bin/openwhisk-standalone.jar --api-gw -p 3233
```

The api gateway will be created on port `3234` by default unless `--api-gw-port` is specified

We also need to modify action creation to support web interface:

```bash
$ wsk action create mnist --docker natrium233/python3action-mnist:1.2 --web true
ok: got action mnist
```

By running `wsk api` command, we enble RESTful access to our action:

```bash
$ wsk api create /ml /mnist post mnist --response-type json
ok: created API /ml/mnist POST for action /_/mnist
http://172.17.0.1:3234/api/23bc46b1-71f6-4ed5-8c54-816aa4f8c502/ml/mnist
```

> If the action is already created without `--web true`, `wsk action update "/_/mnist" --web true` need to be executed to update it

> `172.17.0.1` is docker bridge network IP. However, We can also access to API via `localhost` since API gateway is enabled.

Test this api with curl

```bash
$ curl -X POST -d '{"url":"http://192.168.1.82:9000/mnist/test_picture.png"}' \
       -H 'Content-Type: application/json' \
       http://localhost:3234/api/23bc46b1-71f6-4ed5-8c54-816aa4f8c502/ml/mnist
{
  "code": 200,
  "res": 6
}
```

Once firewall rules are configured, this service will be externally accessiable.  

## Summary

The workflow of using his ML service is:

1. Upload the image to OSS, get access url
2. Invoke OpenWhisk action with url
3. Process with reesult
4. Remove the uploaded image if needed.

A python script is created to accomplish this workflow

```python
"""mnist_demo.py
"""
import argparse
from typing import Union, Dict
from file_uploader import oss_upload
import requests
import json
from minio import Minio
import uuid
import os
from datetime import timedelta


def invoke_openwsk_action_mnist(wsk_apihost: str, image_url: str):
    """Invoke openwsk action via RESTful API

    Args:
        wsk_apihost (str): url like string 
        image_url (str): url to image
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=wsk_apihost,
                             headers=headers,
                             data=json.dumps({"url": image_url}))
    return response.json()


if __name__ == '__main__':
    """Usage

    python file_upload.py --endpoint=192.168.1.82:9000 \
                          --access_key=testAccessKey \
                          --secret_key=testSecretKey \
                          --bucket=mnist \
                          --file=mnist.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str)
    parser.add_argument('--access_key', type=str)
    parser.add_argument('--secret_key', type=str)
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--object_name', type=str, default='')
    parser.add_argument('--mnist_api', type=str)
    args = parser.parse_args()

    minioClient = Minio(args.endpoint,
                        access_key=args.access_key,
                        secret_key=args.secret_key,
                        secure=False)  # Create MinIO client

    if len(args.object_name) <= 0:
        # Generate unique object name
        object_name = str(uuid.uuid1()) + os.path.splitext(args.file)[-1]
    else:
        object_name = args.object_name

    # Upload image
    minioClient.fput_object(args.bucket_name, 
                            object_name,
                            args.file)
    img_url = minioClient.presigned_get_object(args.bucket_name,
                                           object_name,
                                           expires=timedelta(days=2))

    # Invoke OpenWhisk via api
    res = invoke_openwsk_action_mnist(args.mnist_api, img_url)
    print(res)

    # Delete the image
    minioClient.remove_object(args.bucket_name, object_name)
```

We can run the demo by passing arguments:

```bash
$ python mnist_demo.py \
    --endpoint=192.168.1.82:9000 \
    --access_key=testAccessKey \
    --secret_key=testSecretKey \
    --bucket_name=mnist \
    --file=test_picture2.png \
    --mnist_api=http://192.168.1.82:3234/api/23bc46b1-71f6-4ed5-8c54-816aa4f8c502/ml/mnist
{'code': 200, 'res': 4}
```

## Appendix - Convenient setup script

see [ICE6405P-260-M01](https://github.com/davidliyutong/ICE6405P-260-M01/tree/main/scripts/ubuntu/20.04)
