import gc

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from langchain.tools import tool
from PIL import Image
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

from minervaai.utils import create_random_file_name


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
    pipe.enable_model_cpu_offload()
    image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
    del pipe
    gc.collect()
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


def image_understanding(file_path: str, prompt: str) -> str:
    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(file_path),
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


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
