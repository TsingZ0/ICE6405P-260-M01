qemu-system-x86_64 -serial stdio \
-smp 2,sockets=1,cores=2,threads=1 -m 4096 \
-device virtio-gpu-pci \
-display default,show-cursor=on \
-device qemu-xhci -device usb-kbd \
-device usb-tablet -device intel-hda \
-device hda-duplex \
-drive file=centos_disk_dummy.img,if=virtio,cache=writethrough \
-cdrom CentOS-Stream-8-x86_64-latest-boot.iso \
-nic user,model=virtio
