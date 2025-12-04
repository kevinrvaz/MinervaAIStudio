import base64
import shutil
from io import BytesIO
from pathlib import Path

from langchain.tools import tool
from PIL import Image

from minervaai.common import (
    OUTPUTS_PATH,
    THREED_IMAGE,
    app,
    create_random_file_name,
    outputs,
    read_remote_file,
    secrets,
    volumes,
)
from minervaai.tools.images import pil_to_base64

with THREED_IMAGE.imports():
    import sys

    sys.path.insert(0, "/Hunyuan3D-2.1/hy3dshape")
    sys.path.insert(0, "/Hunyuan3D-2.1/hy3dpaint")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
    from textureGenPipeline import ( # type: ignore
        Hunyuan3DPaintConfig,  # type: ignore
        Hunyuan3DPaintPipeline, # type: ignore
    )
    from torchvision_fix import apply_fix  # type: ignore

    apply_fix()


@app.function(
    gpu="h100", image=THREED_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def generate_3d_model_from_image_internal(image_str: str) -> str:
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
    file_name = create_random_file_name("glb", False)
    mesh_textured = paint_pipeline(
        "mesh_untextured.glb", image_path=image, output_mesh_path=file_name
    )
    shutil.copy(file_name, Path(OUTPUTS_PATH) / file_name)
    outputs.commit()
    return mesh_textured


def generate_3d_model_from_image(file_path: str):
    img = Image.open(file_path)
    b64 = pil_to_base64(img, img.format)
    mesh_name = generate_3d_model_from_image_internal.remote(b64)
    return read_remote_file(mesh_name, "glb")


@tool
def generate_3d_object_from_image_tool(image_file_path: str) -> str:
    """Generate a 3D object glb format from a provided image file path

    Args:
        image_file_path (str): The path to the image

    Returns:
        The path to the generated glb file
    """

    return generate_3d_model_from_image(image_file_path)


THREED_TOOLS = {
    "label": "3D Tools",
    "tools": [
        {
            "tool": generate_3d_object_from_image_tool,
            "label": "Generate 3D Object",
            "default": True,
            "tool_id": "generate_3d_object_from_image_tool",
        },
    ],
}
