FROM tensorflow/tensorflow:1.15.5-gpu-py3

# https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112682423
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

COPY . /HT2
WORKDIR /HT2
RUN apt update
# RUN apt install -y python3-tk
RUN python -m pip install -r requirements.txt
CMD "bash"
