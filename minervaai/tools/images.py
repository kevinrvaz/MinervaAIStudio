import base64
from io import BytesIO

from langchain.tools import tool

from minervaai.common import BASE_IMAGE, app, create_random_file_name, secrets, volumes

with BASE_IMAGE.imports():
    import torch
    from diffusers import (
        FluxKontextPipeline,
        FluxPipeline,
        FluxTransformer2DModel,
        GGUFQuantizationConfig,
    )
    from diffusers.utils import export_to_video
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from minervaai.tools.video import common_ltx_pipeline


def pil_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_string


@app.function(
    gpu="h100", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def img_to_video_internal(image: str, prompt: str, negative_prompt: str):
    return common_ltx_pipeline(image, prompt, negative_prompt)


def img_to_video(image: str, prompt: str, negative_prompt: str):
    file_name = create_random_file_name("mp4")
    pil_image = Image.open(image)
    b64 = pil_to_base64(pil_image, pil_image.format)
    video = img_to_video_internal.remote(b64, prompt, negative_prompt)
    export_to_video(video, file_name, fps=24)
    return file_name


@app.function(
    gpu="h100", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def generate_image_internal(prompt: str):
    ckpt_path = (
        "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_1.gguf"
    )
    transformer = FluxTransformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
    byte_stream = BytesIO()
    image.save(byte_stream, format="PNG")
    return byte_stream.getvalue()


@app.function(
    gpu="h100", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def image_editing_internal(image: str, prompt: str):
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    decoded_bytes = base64.b64decode(image)
    image_stream = BytesIO(decoded_bytes)
    pil_image = Image.open(image_stream)
    generated_image = pipe(image=pil_image, prompt=prompt, guidance_scale=2.5).images[0]
    byte_stream = BytesIO()
    generated_image.save(byte_stream, format="PNG")
    return byte_stream.getvalue()


def image_editing(image, prompt):
    file_name = create_random_file_name("png")
    pil_image = Image.open(image)
    b64 = pil_to_base64(pil_image, pil_image.format)
    generated_image = image_editing_internal.remote(b64, prompt)
    generated_image = Image.open(BytesIO(generated_image))
    generated_image.save(file_name)
    return file_name


def generate_image(prompt):
    file_name = create_random_file_name("png")
    generated_image = generate_image_internal.remote(prompt)
    image = Image.open(BytesIO(generated_image))
    image.save(file_name)
    return file_name


def image_resize_to_new_width(file_path: str, new_width: int) -> str:
    image = Image.open(file_path)
    w_percent = new_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    file_name = create_random_file_name("png")
    img = image.resize((new_width, h_size), Image.Resampling.LANCZOS)
    img.save(file_name)
    return file_name


@app.function(
    gpu="h100", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def image_understanding_internal(image: str, prompt: str) -> str:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. Keep responses concise unless requested for a detailed response. All responses must follow a markdown format.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def image_understanding(file_path: str, prompt: str) -> str:
    image = Image.open(file_path)
    b64 = pil_to_base64(image, image.format)
    return image_understanding_internal.remote(b64, prompt)


@tool
def image_resizer_tool(file_path: str, new_width: int) -> str:
    """Resize an image with new width and get back file path, also when using this tool
    do not try to show the image, just return the file path.

    Args:
        file_path (str): The path to the image
        new_width (int): The new width to which the image needs to be resized

    Returns:
        file path of the resized image
    """
    return image_resize_to_new_width(file_path, new_width)


@tool
def image_understanding_tool(file_path: str, prompt: str) -> str:
    """Use this tool to understand whats in the image file path given a prompt,
    after any task which gives back an image file path, use this tool to ask a detailed
    understanding of the image to ensure it follows whatever the user asked.

    Args:
        file_path (str): The file path of the image
        prompt (str): The prompt you want to use with the image

    Returns:
        an answer for prompt for the image
    """
    return image_understanding(file_path, prompt)


@tool
def generate_image_tool(prompt: str) -> str:
    """Generate an image given a prompt and get back a file path, also when using this tool
    do not try to show the image, just return the file path.

    Args:
        prompt (str): The prompt to generate an image.

    Returns:
        file path of generated image
    """
    return generate_image(prompt)


@tool
def image_editing_tool(file_path: str, prompt: str) -> str:
    """Use this tool to edit an image given a file path and the a prompt basis guiding the edit,
    also when using this tool do not try to show the image, just return the file path

    Args:
        file_path (str): The file path of the image
        prompt (str): The prompt you want to use to edit the image

    Returns:
        file path of the edited image
    """
    return image_editing(file_path, prompt)


@tool
def img_to_video_tool(file_path: str, prompt: str, negative_prompt: str) -> str:
    """Generate a video file given a image file path, an input prompt and negative prompt and get back a file path, also when using this tool
    do not try to show the video, just return the file path. if the user doesn't provide a negative prompt then make one
    eg:- worst quality, inconsistent motion, blurry, jittery, distorted

    Args:
        file_path (str): The file path of the image to use for video generation
        prompt (str): The prompt basis which the video should be created
        negative_prompt (str): A negative prompt for what the video shouldn't contain

    Returns:
        file path of the generated video
    """
    return img_to_video(file_path, prompt, negative_prompt)


IMAGE_TOOLS = {
    "label": "Image Tools",
    "tools": [
        {
            "tool": generate_image_tool,
            "label": "Generate Image Tool",
            "default": True,
            "tool_id": "generate_image_tool",
        },
        {
            "tool": image_resizer_tool,
            "label": "Image Resizer",
            "default": True,
            "tool_id": "image_resizer",
        },
        {
            "tool": image_understanding_tool,
            "label": "Image Understanding",
            "default": True,
            "tool_id": "image_understanding_tool",
        },
        {
            "tool": image_editing_tool,
            "label": "Image Editing",
            "default": True,
            "tool_id": "image_editing_tool",
        },
        {
            "tool": img_to_video_tool,
            "label": "Image to Video",
            "default": True,
            "tool_id": "img_to_video_tool",
        },
    ],
}
