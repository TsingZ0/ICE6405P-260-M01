FROM ubuntu:latest
# FROM cuda:10.1

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
    echo "conda activate default" >> ~/.bashrc

ENV PATH=/opt/miniconda/bin:$PATH
COPY ["tsinghua.condarc", "/root/.condarc"]

COPY ["env.yml", "/opt/app/"]
RUN conda env create -f /opt/app/env.yml
RUN . /root/.bashrc && \
    /opt/miniconda/bin/conda init bash && \
    conda activate default && conda info --envs && \
    pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN . /root/.bashrc && \
#     /opt/miniconda/bin/conda init bash && \
#     conda activate default && conda info --envs && \
#     pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
#     pip install cudatoolkit==11.1

COPY ["models.py","worker_utils.py","run_worker.py", "/opt/app/"]
CMD ["/bin/bash", "-c", "cd /opt/app/ && /opt/miniconda/envs/default/bin/python ./run_worker.py"]





