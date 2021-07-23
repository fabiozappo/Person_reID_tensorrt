# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN apt-get update && apt-get install -y screen libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir #to avoid caching

WORKDIR /opt/code/

CMD /bin/bash


