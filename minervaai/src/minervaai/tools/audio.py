import gc
import os

import numpy as np
import soundfile as sf
from kokoro import KPipeline
from langchain.tools import tool

from minervaai.utils import create_random_file_name


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
        }
    ],
}
