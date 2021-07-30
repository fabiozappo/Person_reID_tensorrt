# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt-get update && apt-get install -y screen libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir #to avoid caching

# Install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git /opt/torch2trt
WORKDIR /opt/torch2trt/
RUN git checkout c50039a18060897d933aaf7f6afd3aa541288ec5
RUN python3 setup.py install

WORKDIR /opt/code/

CMD /bin/bash


