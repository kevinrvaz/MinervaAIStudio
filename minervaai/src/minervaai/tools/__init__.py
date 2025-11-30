from minervaai.common import THREED_IMAGE, BASE_IMAGE

with BASE_IMAGE.imports():
    # from minervaai.tools.audio import AUDIO_TOOLS
    from minervaai.tools.general_purpose import GENERAL_TOOLS
    from minervaai.tools.images import IMAGE_TOOLS
    from minervaai.tools.video import VIDEO_TOOLS

with THREED_IMAGE.imports():
    from minervaai.tools.threed import THREED_TOOLS


def filter_tools(tools_selected):
    tools = []
    for modality in (
        GENERAL_TOOLS,
        IMAGE_TOOLS,
        # AUDIO_TOOLS,
        VIDEO_TOOLS,
        THREED_TOOLS,
    ):
        for tool in modality["tools"]:
            if tool["tool_id"] in tools_selected:
                tools.append(tool["tool"])
    return tools


def tool_setter(tool_name, tools, should_keep):
    if should_keep:
        tools.add(tool_name)
    else:
        tools.discard(tool_name)
    return set(tools)


def default_builtin_tool_setter():
    tools = set()
    for modality in (
        GENERAL_TOOLS,
        IMAGE_TOOLS,
        # AUDIO_TOOLS,
        VIDEO_TOOLS,
        THREED_TOOLS,
    ):
        for tool in modality["tools"]:
            if tool["default"]:
                tools.add(tool["tool_id"])
    return tools


__all__ = [
    # "AUDIO_TOOLS",
    "GENERAL_TOOLS",
    "IMAGE_TOOLS",
    "THREED_TOOLS",
    "VIDEO_TOOLS",
    "filter_tools",
    "tool_setter",
    "default_builtin_tool_setter",
]
