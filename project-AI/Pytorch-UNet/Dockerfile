FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD requirements.txt .
RUN python -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
ADD . .
