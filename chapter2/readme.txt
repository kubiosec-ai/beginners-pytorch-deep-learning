docker buildx build --output type=docker --load -t my_pytorch_app .

docker buildx build --platform linux/amd64,linux/arm64  -t my_pytorch_app .



docker run -it -v $(pwd):/app  my_pytorch_app
