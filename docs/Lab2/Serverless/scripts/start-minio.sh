docker run \
    --name minio \
    --net=host \
    -d \
    -v /home/liyutong/Data/ICE6405P-260-M01/minio-data:/data \
    -v /home/liyutong/Data/ICE6405P-260-M01/minio-config:/root/.minio \
    -e "MINIO_ROOT_USER=root" \
    -e "MINIO_ROOT_PASSWORD=password" \
    minio/minio \
    server /data --console-address "0.0.0.0:41309"

docker start minio
