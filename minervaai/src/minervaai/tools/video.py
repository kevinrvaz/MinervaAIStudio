import gc

import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from langchain.tools import tool

from minervaai.utils import create_random_file_name


def text_to_video(prompt, negative_prompt):
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
    )
    pipe.to("mps")
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=256,
        num_frames=161,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        num_inference_steps=50,
    ).frames[0]
    file_name = create_random_file_name("mp4")
    export_to_video(video, file_name, fps=24)
    del pipe
    gc.collect()
    return file_name


@tool
def text_to_video_tool(prompt: str, negative_prompt: str):
    """Generate a video file given an input prompt and negative prompt and get back a file path, also when using this tool
    do not try to show the audio, just return the file path. if the user doesn't provide a negative prompt then make one
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
