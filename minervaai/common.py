import json
import os
import random
import string
from pathlib import Path

import modal

app = modal.App("minervaai-studio")
BASE_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg")
    .entrypoint([])
    .uv_pip_install(
        "gradio==6.0.1",
        "transformers==4.57.1",
        "diffusers==0.35.2",
        "accelerate==1.11.0",
        "sentencepiece==0.2.1",
        "gguf==0.17.1",
        "ftfy==6.3.1",
        "protobuf==6.33.1",
        "pyarrow==22.0.0",
        "ddgs==9.9.1",
        "ollama==0.6.1",
        "langchain==1.0.7",
        "langchain-ollama==1.0.0",
        "langchain-community==0.4.1",
        "torchvision==0.24.1",
        "modal==1.2.4",
        "kokoro>=0.9.2",
        "soundfile==0.13.1",
        "ffmpeg-python==0.2.0",
        "imageio-ffmpeg==0.6.0",
        "fastapi[standard]",
        "bitsandbytes",
        "jinja2",
        "langchainhub",
        "numexpr",
        "text-generation",
        "langchain-huggingface",
        "scipy",
        "langchain-mcp-adapters",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

THREED_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .apt_install("clang")
    .apt_install("wget")
    .run_commands("git clone https://github.com/kevinrvaz/Hunyuan3D-2.1.git")
    .workdir("Hunyuan3D-2.1/")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "/.uv/uv pip install --python $(command -v python) --compile-bytecode -r requirements.txt"
    )
    .workdir("hy3dpaint/custom_rasterizer/")
    .apt_install(
        [
            "libgl1-mesa-glx",
            "ibglib2.0-0",
            "unzip",
            "git-lfs",
            "pkg-config",
            "libglvnd0",
            "libgl1",
            "libglx0",
            "libegl1",
            "libgles2",
            "libglvnd-dev",
            "libgl1-mesa-dev",
            "libegl1-mesa-dev",
            "libgles2-mesa-dev",
            "cmake",
            "curl",
            "mesa-utils-extra",
            "libxrender1",
            "libeigen3-dev",
            "python3-dev",
            "python3-setuptools",
            "libcgal-dev",
            "build-essential",
        ]
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "${CUDA_HOME}/bin:${PATH}",
            "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    .run_commands(
        "/.uv/uv pip install --python /usr/local/bin/python wheel && /.uv/uv pip install --python /usr/local/bin/python -e . --no-build-isolation",
        env={
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0",
            "CUDA_NVCC_FLAGS": "-allow-unsupported-compiler",
        },
    )
    .workdir("../../")
    .workdir("hy3dpaint/DifferentiableRenderer")
    .run_commands("bash compile_mesh_painter.sh")
    .workdir("../..")
    .run_commands(
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt"
    )
    .workdir("/Hunyuan3D-2.1/")
    .apt_install("libxi6", "libxkbcommon0", "libsm6", "libxext6", "libxrender-dev")
    .uv_pip_install(
        "bpy==3.6.0",
        "hf_xet",
        extra_index_url="https://download.blender.org/pypi/",
    )
    .uv_pip_install("ddgs==9.9.1", "langchain==1.0.7", "langchain-mcp-adapters")
    .entrypoint([])
)

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
VOLUME_NAME = "outputs-3d"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
volumes = {CACHE_DIR: cache_volume, OUTPUTS_PATH: outputs}
secrets = [modal.Secret.from_name("huggingface-secret")]


def create_random_file_name(ext, with_suffix=True):
    file = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    if with_suffix:
        return os.path.join("generated_assets", f"{file}.{ext}")
    return f"{file}.{ext}"


def read_mcp_config():
    with open(os.path.join("config", "mcp_config.json")) as file:
        mcp_config = json.load(file)
        return mcp_config
    return {}


def read_remote_file(file_name, file_ext):
    file_name = create_random_file_name(file_ext)
    data = b""
    for chunk in outputs.read_file(file_name):
        data += chunk

    with open(file_name, "wb") as file:
        file.write(data)

    return file_name
