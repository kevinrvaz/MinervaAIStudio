import base64
import gc
from io import BytesIO

import modal
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from langchain.tools import tool
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from minervaai.utils import app, create_random_file_name, image


def pil_to_base64(pil_image, format="PNG"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_string


def generate_image(prompt: str) -> str:
    ckpt_path = (
        "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
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
    )
    image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
    file_name = create_random_file_name("png")
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


@app.function(gpu="h100", image=image)
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
                    "text": "You are a helpful assistant. Keep responses concise unless requested for a detailed response.",
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
    """Use this tool to understand whats in the image file path given a prompt

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
    ],
}
