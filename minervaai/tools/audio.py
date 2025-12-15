import os
import shutil
from pathlib import Path

import torch

from minervaai.common import (
    BASE_IMAGE,
    OUTPUTS_PATH,
    app,
    outputs,
    read_remote_file,
    secrets,
    volumes,
)

with BASE_IMAGE.imports():
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    import scipy

from langchain.tools import tool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from minervaai.common import create_random_file_name


@app.function(
    gpu="T4", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def text_to_speech_internal(text):
    pipeline = KPipeline(lang_code="a", device="cuda")
    generator = pipeline(text, voice="af_heart")
    file_name = create_random_file_name("wav", False)
    for _, _, audio in generator:
        data = np.asarray(audio)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        if os.path.exists(file_name):
            with sf.SoundFile(file_name, "r+") as sound:
                sound.seek(0, sf.SEEK_END)
                sound.write(audio)
        else:
            with sf.SoundFile(
                file_name, "w", samplerate=24000, channels=channels
            ) as sound:
                sound.write(audio)

    shutil.copy(file_name, Path(OUTPUTS_PATH) / file_name)
    outputs.commit()
    return file_name


def text_to_speech(text):
    generated_speech = text_to_speech_internal.remote(text)
    return read_remote_file(generated_speech, "wav")


@app.function(
    gpu="T4", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def speech_to_text_internal(sound_file) -> str:
    with open("sound.wav", "wb") as file:
        file.write(sound_file)

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float32,
        device="cuda",
    )

    res = pipe("sound.wav")
    return res["text"]


def speech_to_text(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_contents = file.read()
    return speech_to_text_internal.remote(file_contents)


@app.function(
    gpu="H100", image=BASE_IMAGE, volumes=volumes, secrets=secrets, timeout=1200
)
def music_generation_internal(prompt):
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")
    music = synthesiser(prompt, forward_params={"do_sample": True})
    file_name = create_random_file_name("wav", False)
    scipy.io.wavfile.write(file_name, rate=music["sampling_rate"], data=music["audio"])
    shutil.copy(file_name, Path(OUTPUTS_PATH) / file_name)
    outputs.commit()
    return file_name


def music_generation(prompt):
    generated_music = music_generation_internal.remote(prompt)
    return read_remote_file(generated_music, "wav")


@tool
def music_generation_tool(prompt: str) -> str:
    """Generate a music file given an input text prompt and get back a file path, also when using this tool
    do not try to show the music, just return the file path.

    Args:
        text (str): text prompt to guide music generation

    Returns:
        file path to the generated music file.
    """
    return music_generation(prompt)


@tool
def speech_to_text_tool(file_path: str) -> str:
    """Convert the audio recording file provided into text and return it.
    you can also use it for understanding audio files. If user only provides an audio
    file and no instruction assume the instruction is part of the audio file and you need to
    use this to get a transcript of the audio file to understand what to do next.

    Args:
        file_path (str): The location of the audio file to be converted to text.

    Returns:
        Text in the audio file provided.
    """
    return speech_to_text(file_path)


@tool
def text_to_speech_tool(text: str) -> str:
    """Generate an audio file given an input text and get back a file path, also when using this tool
    do not try to show the audio, just return the file path.

    Args:
        text (str): text to convert to speech.

    Returns:
        file path to the generated speech file.
    """
    return text_to_speech(text)


AUDIO_TOOLS = {
    "label": "Audio Tools",
    "tools": [
        {
            "tool": text_to_speech_tool,
            "label": "Text to Speech Tool",
            "default": True,
            "tool_id": "text_to_speech_tool",
        },
        {
            "tool": speech_to_text_tool,
            "label": "Speech to Text Tool",
            "default": True,
            "tool_id": "speech_to_text_tool",
        },
        {
            "tool": music_generation_tool,
            "label": "Music Generation",
            "default": True,
            "tool_id": "music_generation_tool",
        },
    ],
}
