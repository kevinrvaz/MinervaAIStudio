import torch
from langchain.tools import tool
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.utils import export_to_video

from minervaai.utils import app, create_random_file_name, image, secrets, volumes


@app.function(gpu="h100", image=image, volumes=volumes, secrets=secrets, timeout=1200)
def text_to_video_internal(prompt, negative_prompt):
    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16
    )
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7",
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe_upsample.to("cuda")
    pipe.vae.enable_tiling()

    def round_to_nearest_resolution_acceptable_by_vae(height, width):
        height = height - (height % pipe.vae_spatial_compression_ratio)
        width = width - (width % pipe.vae_spatial_compression_ratio)
        return height, width

    expected_height, expected_width = 512, 704
    downscale_factor = 2 / 3
    num_frames = 121

    downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(
        expected_width * downscale_factor
    )
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width
    )
    latents = pipe(
        conditions=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",
    ).frames

    upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
    upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=upscaled_width,
        height=upscaled_height,
        num_frames=num_frames,
        denoise_strength=0.4,
        num_inference_steps=10,
        latents=upscaled_latents,
        decode_timestep=0.05,
        image_cond_noise_scale=0.025,
        generator=torch.Generator().manual_seed(0),
        output_type="pil",
    ).frames[0]

    video = [frame.resize((expected_width, expected_height)) for frame in video]
    return video


def text_to_video(prompt, negative_prompt):
    file_name = create_random_file_name("mp4")
    video = text_to_video_internal.remote(prompt, negative_prompt)
    export_to_video(video, file_name, fps=24)
    return file_name


@tool
def text_to_video_tool(prompt: str, negative_prompt: str):
    """Generate a video file given an input prompt and negative prompt and get back a file path, also when using this tool
    do not try to show the video, just return the file path. if the user doesn't provide a negative prompt then make one
    eg:- worst quality, inconsistent motion, blurry, jittery, distorted

    Args:
        prompt (str): The prompt basis which the video should be created
        negative_prompt (str): A negative prompt for what the video shouldn't contain
    """
    return text_to_video(prompt, negative_prompt)


VIDEO_TOOLS = {
    "label": "Video Tools",
    "tools": [
        {
            "tool": text_to_video_tool,
            "label": "Text to Video Tool",
            "default": True,
            "tool_id": "text_to_video_tool",
        }
    ],
}
