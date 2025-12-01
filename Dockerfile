FROM docker.m.daocloud.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Replace APT sources with Aliyun mirrors for faster download in China
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list

# Install Python 3.11 and build dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common curl git build-essential && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# Configure pip to use Tsinghua mirror (fastest for China)
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install taichi==1.7.4 numpy pyyaml matplotlib

# Set working directory
WORKDIR /app

# Default command
CMD ["/bin/bash"]
