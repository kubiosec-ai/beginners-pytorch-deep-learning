# Building multi-arch docker images

### 1. Create Ubuntu server
### 2. Install docker
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```
```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
### 3. Enable containerd store
Create `/etc/docker/daemon.json`
```
{
  "features": {
    "containerd-snapshotter": true
  }
}
```
```
sudo systemctl restart docker
```
See https://docs.docker.com/storage/containerd/ for more info
### 4. Build the images 
```
docker buildx build --push --platform linux/arm64,linux/amd64  -t xxradar/my_pytorch_app .
```

### 5. Test the image
Just out of convenience ...
```
docker run -it -v $(pwd):/app  -v $(pwd)/../data:/data -v $(pwd)/../model:/model xxradar/my_pytorch_app
```
