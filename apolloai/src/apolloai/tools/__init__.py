from apolloai.tools.audio import AUDIO_TOOLS
from apolloai.tools.general_purpose import GENERAL_TOOLS
from apolloai.tools.images import IMAGE_TOOLS
from apolloai.tools.threed import THREED_TOOLS
from apolloai.tools.video import VIDEO_TOOLS

def filter_tools(tools_selected):
    tools = []
    for modality in (
        GENERAL_TOOLS,
        IMAGE_TOOLS,
        AUDIO_TOOLS,
        VIDEO_TOOLS,
        THREED_TOOLS,
    ):
        for tool in modality["tools"]:
            if tool["tool_id"] in tools_selected:
                tools.append(tool["tool"])
    return tools

__all__ = [
    "AUDIO_TOOLS",
    "GENERAL_TOOLS",
    "IMAGE_TOOLS",
    "THREED_TOOLS",
    "VIDEO_TOOLS",
    "filter_tools"
]