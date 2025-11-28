import gc
import os

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from langchain.tools import tool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from minervaai.utils import create_random_file_name
import scipy

def text_to_speech(text):
    pipeline = KPipeline(lang_code="a", device="mps")
    generator = pipeline(text, voice="af_heart")
    file_name = create_random_file_name("wav")
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

    del generator
    del pipeline
    gc.collect()
    return file_name


def speech_to_text(file_path: str) -> str:
    device = "mps" if torch.mps.is_available() else "cpu"
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float32,
        device=device,
    )

    res = pipe(file_path)
    del pipe
    gc.collect()
    return res["text"]


def music_generation(prompt):
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")
    music = synthesiser(prompt, forward_params={"do_sample": True})
    file_name = create_random_file_name("wav")
    scipy.io.wavfile.write(file_name, rate=music["sampling_rate"], data=music["audio"])
    del synthesiser
    gc.collect()
    return file_name

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
    you can also use it for understanding audio files.

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
