# Contents of this folder

During experiment, the content of folder is as follows

```text
./docs/Lab1/scripts
├── CentOS-Stream-8-x86_64-latest-boot.iso -> .../CentOS-Stream-8-x86_64-latest-boot.iso
├── README.md
├── centos_disk_0.img -> .../centos_disk_0.img
├── centos_disk_1.img -> .../centos_disk_0.img (hard link)
├── dpdk-simple-start.sh
├── ovs-bind-port.sh
├── ovs-simple-start.sh
├── ovs-simple-stop.sh
├── qemu-install-os.sh
├── qemu-kvm-start.sh
├── qemu-migrate-1-start.sh
├── qemu-migrate-2-start.sh
├── qemu-multiqueue-start.sh
└── qemu-simple-start.sh
```

- `centos_disk_0.img` Qemu disk image with OS
- `centos_disk_dummy.img` Empty disk image
- `CentOS-Stream-8-x86_64-latest-boot.iso` Installer
