FROM ubuntu:latest

ENV LANG=C.UTF-8

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh && \
    /opt/miniconda/bin/conda clean -tipsy && \
    ln -s /opt/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate mnist" >> ~/.bashrc

ENV PATH=/opt/miniconda/bin:$PATH PATH_TO_STATE_DICT=/opt/mnist/model.pth
COPY ["tsinghua.condarc", "/root/.condarc"]
COPY ["env.yml", "/opt/mnist/"]
RUN conda env create -f /opt/mnist/env.yml
RUN . /root/.bashrc && \
    /opt/miniconda/bin/conda init bash && \
    conda activate mnist && conda info --envs && \
    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY ["model.pth","model.py","deploy-flask.py", "/opt/mnist/"]
CMD ["/bin/bash", "-c", "/opt/miniconda/envs/mnist/bin/python /opt/mnist/deploy-flask.py"]





