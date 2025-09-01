### Installation
1.Docker pull

```python
docker pull crpi-oe9af960c637cs3a.cn-hangzhou.personal.cr.aliyuncs.com/event_camera/ev-uav:v1.0
```
2.Install **nvidia-container-runtime** for auto

```python
sudo apt update
sudo apt install -y nvidia-container-runtime
```

Issues that may arise during installation, please refer to https://blog.csdn.net/weixin_52582300/article/details/146316282.

3.Operating container

```python
docker run --runtime=nvidia --gpus all -it -d --name evuav crpi-oe9af960c637cs3a.cn-hangzhou.personal.cr.aliyuncs.com/event_camera/ev-uav:v1.0 bash
```
4.Entering the container

```python
docker exec -it evuav bash
```
5.Running code

```python
conda activate evuav
cd /root/EV-UAV/ && python train.py
```