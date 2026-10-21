# Pin the CUDA base tag to the CUDA version your torch build and the TargetDiff repo expect (verify first).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
# 1) torch from the official index for this exact CUDA (edit versions to match docs/VERSIONS.md)
RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# 2) PyG + extensions from the official wheel index for that exact torch/CUDA pair
RUN pip install torch_geometric && pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/workspace/src
CMD ["python3", "-m", "pytest", "-q"]
