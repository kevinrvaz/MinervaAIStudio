from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
import random
import string


def generate_3d_mesh_from_image(file_path: str) -> str:
    image = Image.open(file_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="mps")
    mesh = pipeline(image=image)[0]
    file = "".join(random.choices(string.ascii_letters, k=10))
    file_name = f'{file}.glb'
    mesh.export(file_name)
    return file_name

def generate_3d_model_from_image(file_path: str) -> str:
    image = Image.open(file_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="mps")
    mesh = pipeline(image=image)[0]
    
    pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    pipeline.enable_model_cpu_offload()
    mesh = pipeline(mesh, image=image)
    file = "".join(random.choices(string.ascii_letters, k=10))
    file_name = f'{file}.glb'
    mesh.export(file_name)
    return file_name


THREED_TOOLS = {
    "label": "3D Tools",
    "tools": []
}