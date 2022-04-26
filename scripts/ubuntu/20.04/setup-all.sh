#!/bin/bash

set -e

bash setup-deps.sh
bash setup-qemu.sh
bash setup-dpdk.sh
bash setup-ovs.sh
bash setup-openwsk.sh
# In silient installation (running with sudo, for example) the 
# docker command will always be available even if current user 
# is not in then `docker` group
bash setup-docker.sh
