API_HOST=http://localhost:8080/run
PICTURE_URL=http://192.168.1.82:9000/mnist/test_picture.png
curl -X POST -d '{"value":{"url":"'$PICTURE_URL'"}}' -H 'Content-Type: application/json' $API_HOST