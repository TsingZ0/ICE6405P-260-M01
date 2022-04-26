# Install script for QEMU, DPDK, OpenVSwitch, OpenWhisk

For convinence, you can use these script to setup QEMU, DPDK, OpenVSwitch and OpenWhisk

## Tested environment

The scripts are tested on Ubuntu 20.04 LTS, amd64 architecture with `bash`. It might be applicable for Debian distributions.

For CentOS/Fedora distributions that use `yum` package manager, you should modify `setup-deps.sh` to use yum

Not tested on ARM architecture.

The scripts will write to some directories. The might **OVERWRITTEN YOUR IMPORTANT FILES** in case of a wrong configuration.

## Overview & Usage

There are 5 scripts:

- `setup-deps.sh` Script for dependencies
- `setup-qemu.sh` Script for QEMU
- `setup-dpdk.sh` Script for DPDK
- `setup-ovs.sh` Script for OpenVSwitch
- `setup-openwsk.sh` Script for OpenWhisk (java standalone)
- `setup-docker.sh` Script for docker (see [Get Docker](https://docs.docker.com/get-docker/))
They can be run with

```bash
bash $SCRIPT
```

You should always **exam the shell script** carefully.

Take `setup-qemu.sh` as example:

```bash
#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
SRC_PATH=~/Src/qemu
QEMU_VERSION=6.1.0
PROFILE=~/.bashrc
# -------- END CONFIGURATION -------- #

CURR_PATH=$(pwd)
N_PROC=$(expr $(cat /proc/cpuinfo |grep "processor"|wc -l) \* 2)
set -e
...
```

The lines between `BEGIN CONFIGURATION` and `END CONFIGURATION` are editable

- `SRC_PATH` is the path to store source code and compiled binaries. This path **might be overwritten**.
- `PROFILE` is shell profile, for example `~/.bashrc`. The profile **might be modified**. If you dont want your profile modified, set it to `/dev/null` or remove the line.
- `QEMU_VERSION` is the version of QEMU. The version has to be valid
- `N_PROC` is number of make processes (`make -j$N_PROC`)

### Note

> Some script may modify shell profile and `$PATH` variable. In this case, `source $PROFILE` should be executed to update shell environment.
> Some script may require root priviledge at some stage of execution.
> `setup-openwsk.sh` build standalone jar of openwhisk. It **will not install docker** nor deploy openwsk to k8s.
