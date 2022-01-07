SHARED_PATH=.
sudo -E $(which qemu-system-x86_64) \
   -smp 2,sockets=1,cores=2,threads=1 -m 4096 \
   -device virtio-gpu-pci \
   -display default,show-cursor=on \
   -device qemu-xhci -device usb-kbd \
   -device usb-tablet -device intel-hda \
   -device hda-duplex \
   -drive file=$SHARED_PATH/centos_disk_1.img,if=virtio,cache=writethrough \
   -nic user,model=virtio,hostfwd=tcp::10123-:22 \
   -enable-kvm \
   -incoming tcp:0:16666