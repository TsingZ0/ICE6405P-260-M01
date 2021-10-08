#!/bin/bash
SRC_PATH=~/Src/qemu
QEMU_VERSION=6.1.0
PROFILE=~/.bashrc

CURR_PATH=$(pwd)
N_PROC=$(expr $(cat /proc/cpuinfo |grep "processor"|wc -l) \* 2)
set -e

if [ ! -d "$SRC_PATH" ]; then
echo "making dir $SRC_PATH" && mkdir -p "$SRC_PATH"
fi

cd "$SRC_PATH"

# Download qemu source code
if [ ! -f "qemu-$QEMU_VERSION.tar.xz" ]; then
wget https://download.qemu.org/qemu-$QEMU_VERSION.tar.xz
fi

if [ ! -f "qemu-$QEMU_VERSION" ]; then
tar xvJf qemu-$QEMU_VERSION.tar.xz
fi

cd qemu-$QEMU_VERSION
./configure --enable-vhost-user --enable-vhost-net --enable-kvm  --enable-libusb
make -j$N_PROC
echo "export PATH=$(pwd)/build:\$PATH" >> $PROFILE

echo "Installation completed, you should run 'source $PROFILE' to use qemu commands"
cd $CURR_PATH
