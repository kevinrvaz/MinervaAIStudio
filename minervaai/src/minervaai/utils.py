import os
import random
import string
from pathlib import Path

import modal

app = modal.App("minervaai-studio")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
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
        "hy3dgen==2.0.2",
        "torchvision==0.24.1",
        "modal==1.2.4",
        "kokoro>=0.9.2",
        "soundfile==0.13.1",
        "ffmpeg-python==0.2.0",
        "imageio-ffmpeg==0.6.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}
secrets = [modal.Secret.from_name("huggingface-secret")]


def create_random_file_name(ext):
    file = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    return os.path.join("generated_assets", f"{file}.{ext}")
