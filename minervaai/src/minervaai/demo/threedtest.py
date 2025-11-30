import modal

from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
import random
import string
import shutil
import os


def pil_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_string


app = modal.App("minervaai-studio-3d-test")
CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]
VOLUME_NAME = "outputs-3d"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
volumes = {CACHE_DIR: cache_volume, OUTPUTS_PATH: outputs}


def create_random_file_name(ext):
    file = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    return os.path.join("generated_assets", f"{file}.{ext}")


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
    .run_commands("ls -lha")
    .run_commands("pwd")
    .run_commands(
        "/.uv/uv pip install --python $(command -v python) --compile-bytecode -r requirements.txt"
    )
    .workdir("hy3dpaint/custom_rasterizer/")
    .run_commands("ls")
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
            # "LD_LIBRARY_PATH": "${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}",
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
    .run_commands("echo $LD_LIBRARY_PATH")
    .uv_pip_install(
        "bpy==3.6.0", "hf_xet", extra_index_url="https://download.blender.org/pypi/"
    )
    .entrypoint([])
)

with THREED_IMAGE.imports():
    import sys

    sys.path.insert(0, "/Hunyuan3D-2.1/hy3dshape")
    sys.path.insert(0, "/Hunyuan3D-2.1/hy3dpaint")
    from torchvision_fix import apply_fix  # type: ignore
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # type: ignore
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore

    apply_fix()


@app.function(
    gpu="h100", image=THREED_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def generate_3d_model_from_image(image_str: str) -> str:
    link_path_str = "/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    decoded_bytes = base64.b64decode(image_str)
    image_stream = BytesIO(decoded_bytes)
    image = Image.open(image_stream).convert("RGBA")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1"
    )
    mesh_untextured = shape_pipeline(image=image)[0]
    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_config.realesrgan_ckpt_path = link_path_str
    paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
    mesh_untextured.export("mesh_untextured.glb")
    mesh_textured = paint_pipeline(
        "mesh_untextured.glb", image_path=image, output_mesh_path="mesh_textured.glb"
    )
    shutil.copy("mesh_textured.glb", Path(OUTPUTS_PATH) / "mesh_textured.glb")
    outputs.commit()
    return mesh_textured


@app.local_entrypoint()
def threed_test():
    img_path = "/Users/kevinvaz/SoftwareProjects/ApolloAI/minervaai/generated_assets/tu6KgrmShPaTf46bX3ST.png"
    img = Image.open(img_path)
    b64 = pil_to_base64(img, img.format)
    mesh_name = generate_3d_model_from_image.remote(b64)
    print(mesh_name)
