# Remote GPU Setup

This feature allows you to run the face swapping models on a remote GPU server.

## Setup

```
docker run --rm -it -v$PWD:/ctx -w/ctx -p 9999:9999 --entrypoint bash --gpus 1 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
apt update && apt install -y ffmpeg libsm6 libxext6 git python3 python3-pip wget
wget https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true -O models/inswapper_128_fp16.onnx
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
```

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Start Remote GPU Server

On the machine with GPU:

```bash
python3 remote_gpu_server.py
```

The server will start on `http://0.0.0.0:9999`

### 3. Run Client with Remote GPU

```bash
python3 run.py --remote-gpu --remote-gpu-url http://YOUR_GPU_SERVER_IP:9999
```

## Options

- `--remote-gpu`: Enable remote GPU processing
- `--remote-gpu-url`: URL of the remote GPU server (default: http://localhost:9999)

## Notes

- The remote server needs the `models/inswapper_128_fp16.onnx` file
- If remote GPU is unavailable, it will fallback to local processing
- Network latency will affect processing speed
