import os
from uuid import uuid4

import gradio as gr

from apolloai.llm import chat_completion, history_builder, llm_config
from apolloai.tools import *
from apolloai.tools.general_purpose import search_tool
from apolloai.tools.images import generate_image, image_resize_to_new_width
from apolloai.tools.threed import (
    generate_3d_mesh_from_image,
    generate_3d_model_from_image,
)

chats = [{"session_id": uuid4(), "messages": [], "short_name": "Dummy Session"}]
main_page = "Agent Mode"


def set_parameter_field(parameter, val, config):
    return {**config, parameter: val}


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
        AUDIO_TOOLS,
        VIDEO_TOOLS,
        THREED_TOOLS,
    ):
        for tool in modality["tools"]:
            if tool["default"]:
                tools.add(tool["tool_id"])
    return tools


with gr.Blocks(
    fill_height=True,
    title="Apollo AI Studio"
) as demo:
    model_config_state = gr.State({})
    model_selected = gr.State("gpt-oss:20b")
    builtin_tools_selected = gr.State(default_builtin_tool_setter())

    navbar = gr.Navbar(visible=True, main_page_name=main_page)
    with gr.Sidebar():
        gr.Markdown("""
        # Apollo AI Studio
        Apollo AI is an AI assistant that can help in creative tasks like creating and editing images, video, audio and 3d structures using open weight models.
        ## Chat Sessions
        """)
        html_string = ""
        for message in chats:
            html_string += f"<h3>{message['short_name']}</h3>"
        print(html_string)
        gr.Markdown(f" <hr/>{html_string} <hr/>")

    @gr.on(
        triggers=[model_selected.change, demo.load],
        inputs=[model_selected],
        outputs=[model_config_state],
    )
    def default_model_config_setter(model_selected):
        default_params = {}
        if model_selected not in llm_config:
            return default_params
        for parameter in sorted(
            llm_config[model_selected]["inference_parameters"].keys(),
            key=lambda k: llm_config[model_selected]["inference_parameters"][k][
                "order"
            ],
        ):
            default_params[parameter] = llm_config[model_selected][
                "inference_parameters"
            ][parameter]["default"]
        return default_params

    def build_dynamic_model_settings(model_selected):
        blocks = [
            {"field": gr.Markdown(f"## {model_selected} configuration", render=False)}
        ]

        if model_selected not in llm_config:
            return blocks + [{"field": gr.Markdown("# Config not found", render=False)}]

        for parameter in sorted(
            llm_config[model_selected]["inference_parameters"].keys(),
            key=lambda k: llm_config[model_selected]["inference_parameters"][k][
                "order"
            ],
        ):
            if (
                llm_config[model_selected]["inference_parameters"][parameter]["type"]
                == "slider"
            ):
                blocks.append(
                    {
                        "field": gr.Slider(
                            maximum=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["max"],
                            minimum=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["min"],
                            value=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            label=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                            interactive=True,
                            render=False,
                        ),
                    }
                )
                blocks[-1]["field"].change(
                    set_parameter_field,
                    inputs=[
                        gr.State(parameter),
                        blocks[-1]["field"],
                        model_config_state,
                    ],
                    outputs=[model_config_state],
                )
            elif (
                llm_config[model_selected]["inference_parameters"][parameter]["type"]
                == "dropdown"
            ):
                blocks.append(
                    {
                        "field": gr.Dropdown(
                            render=False,
                            interactive=True,
                            value=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            choices=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["options"],
                            label=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                        ),
                    }
                )
                blocks[-1]["field"].select(
                    set_parameter_field,
                    inputs=[
                        gr.State(parameter),
                        blocks[-1]["field"],
                        model_config_state,
                    ],
                    outputs=[model_config_state],
                )
            elif (
                llm_config[model_selected]["inference_parameters"][parameter]["type"]
                == "number"
            ):
                blocks.append(
                    {
                        "field": gr.Number(
                            maximum=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["max"],
                            minimum=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["min"],
                            value=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            label=llm_config[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                            interactive=True,
                            render=False,
                        ),
                    }
                )
                blocks[-1]["field"].change(
                    set_parameter_field,
                    inputs=[
                        gr.State(parameter),
                        blocks[-1]["field"],
                        model_config_state,
                    ],
                    outputs=[model_config_state],
                )
            else:
                print(
                    f"Unsupported parameter config {llm_config[model_selected]["inference_parameters"][parameter]}"
                )
                break

        return blocks

    @gr.render(triggers=[model_selected.change, demo.load], inputs=[model_selected])
    def get_agent_settings_sidebar(model_selected):
        with gr.Sidebar(position="right", open=False, width=360):
            gr.Markdown("""
            # Agent Settings
            ## Model Settings
            """)
            for block in build_dynamic_model_settings(model_selected):
                block["field"].render()

            gr.Markdown("## Agent Tooling")
            with gr.Accordion(label="Builtin Tools", open=False):
                with gr.Accordion(label=GENERAL_TOOLS["label"], open=False):
                    for tool in GENERAL_TOOLS["tools"]:
                        box = gr.Checkbox(
                            value=tool["default"], label=tool["label"], interactive=True
                        )
                        box.input(
                            tool_setter,
                            inputs=[
                                gr.State(tool["tool_id"]),
                                builtin_tools_selected,
                                box,
                            ],
                            outputs=[builtin_tools_selected],
                        )

                with gr.Accordion(label=IMAGE_TOOLS["label"], open=False):
                    for tool in IMAGE_TOOLS["tools"]:
                        box = gr.Checkbox(
                            value=tool["default"], label=tool["label"], interactive=True
                        )
                        box.input(
                            tool_setter,
                            inputs=[
                                gr.State(tool["tool_id"]),
                                builtin_tools_selected,
                                box,
                            ],
                            outputs=[builtin_tools_selected],
                        )

                with gr.Accordion(label=VIDEO_TOOLS["label"], open=False):
                    for tool in VIDEO_TOOLS["tools"]:
                        box = gr.Checkbox(
                            value=tool["default"], label=tool["label"], interactive=True
                        )
                        box.input(
                            tool_setter,
                            inputs=[
                                gr.State(tool["tool_id"]),
                                builtin_tools_selected,
                                box,
                            ],
                            outputs=[builtin_tools_selected],
                        )

                with gr.Accordion(label=AUDIO_TOOLS["label"], open=False):
                    for tool in AUDIO_TOOLS["tools"]:
                        box = gr.Checkbox(
                            value=tool["default"], label=tool["label"], interactive=True
                        )
                        box.input(
                            tool_setter,
                            inputs=[
                                gr.State(tool["tool_id"]),
                                builtin_tools_selected,
                                box,
                            ],
                            outputs=[builtin_tools_selected],
                        )

                with gr.Accordion(label=THREED_TOOLS["label"], open=False):
                    for tool in THREED_TOOLS["tools"]:
                        box = gr.Checkbox(
                            value=tool["default"], label=tool["label"], interactive=True
                        )
                        box.input(
                            tool_setter,
                            inputs=[
                                gr.State(tool["tool_id"]),
                                builtin_tools_selected,
                                box,
                            ],
                            outputs=[builtin_tools_selected],
                        )

            with gr.Accordion(label="MCP Servers", open=False):
                gr.Markdown("# TODO")

            gr.Markdown("## Agents")
            with gr.Accordion(label="Agents", open=False):
                pass

    chatbot = gr.Chatbot(
        avatar_images=(
            os.path.join("src", "apolloai", "images", "user.png"),
            os.path.join("src", "apolloai", "images", "owl.png"),
        ),
        show_label=False,
        elem_id="chatbot-section",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
            text_field = gr.MultimodalTextbox(
                sources=["upload", "microphone"],
                interactive=True,
                show_label=False,
                autofocus=True,
                placeholder="Enter a message to Apollo.",
                stop_btn=True,
                elem_id="InputField",
            )
            text_field.submit(
                history_builder,
                [chatbot, text_field],
                [chatbot, text_field],
                queue=False,
            ).success(
                chat_completion,
                inputs=[
                    chatbot,
                    model_selected,
                    model_config_state,
                    builtin_tools_selected,
                ],
                outputs=chatbot,
            )

        with gr.Column(scale=4):
            model_selector = gr.Dropdown(
                ["gpt-oss:20b", "claude-sonnet-4.5", "gpt-5"],
                label="Model Selector",
                interactive=True,
            )
            model_selector.change(
                lambda val: val, inputs=[model_selector], outputs=[model_selected]
            )

with demo.route("Agent Creator"):
    pass

with demo.route("Builtin Tools"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)
    with gr.Tab("General Purpose"):
        with gr.Tab("Web Search"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.TextArea(label="Search Query", interactive=True)
                    search_button = gr.Button("Search")
                with gr.Column():
                    search_results = gr.TextArea(
                        interactive=False, label="Search Results"
                    )
                search_button.click(
                    search_tool.invoke, inputs=[search_query], outputs=[search_results]
                )

    with gr.Tab("Image"):
        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column():
                    image_query = gr.TextArea(label="Image Prompt", interactive=True)
                    image_button = gr.Button("Generate Image")
                with gr.Column():
                    image_result = gr.Image(interactive=False, label="Generated Image")
                image_button.click(
                    generate_image, inputs=[image_query], outputs=[image_result]
                )

        with gr.Tab("Image Resizer"):
            with gr.Row():
                with gr.Column():
                    new_width = gr.Number(label="New width", interactive=True)
                    image_to_resize = gr.Image(label="Image to resize", type="filepath")
                    resizer_button = gr.Button("Resize")
                with gr.Column():
                    image_result = gr.Image(interactive=False, label="Resized Image")
                resizer_button.click(
                    image_resize_to_new_width,
                    inputs=[image_to_resize, new_width],
                    outputs=[image_result],
                )

    with gr.Tab("Video"):
        pass

    with gr.Tab("Audio"):
        pass

    with gr.Tab("3D"):
        with gr.Tab("Image to 3D Mesh"):
            with gr.Column():
                image_to_3d = gr.Image(
                    label="Image to create 3d mesh from", type="filepath"
                )
                render_button = gr.Button("Generate 3D Mesh")
            with gr.Column():
                model_3d = gr.Model3D(interactive=False, label="Generated 3D Mesh")
            render_button.click(
                generate_3d_mesh_from_image, inputs=[image_to_3d], outputs=[model_3d]
            )

        with gr.Tab("Image to 3D Model"):
            with gr.Column():
                image_to_3d = gr.Image(
                    label="Image to create 3d model from", type="filepath"
                )
                render_button = gr.Button("Generate 3D Model")
            with gr.Column():
                model_3d = gr.Model3D(interactive=False, label="Generated 3D Model")
            render_button.click(
                generate_3d_model_from_image, inputs=[image_to_3d], outputs=[model_3d]
            )

with demo.route("MCP Servers"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)

with demo.route("Model Manager"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)

    with gr.Tab("Model Inference Provider"):
        choices = ["ollama", "system", "modal"]
        gr.Markdown("# Text Generation & Reasoning Models")
        with gr.Row():
            gr.Dropdown(choices=choices, label="gpt-oss:20b", interactive=True)
            gr.Dropdown(choices=choices, label="gpt-oss:120b", interactive=True)

        gr.Markdown("# Image Models")
        with gr.Row():
            gr.Dropdown(choices=choices, label="FLUX.1-dev", interactive=True)
            gr.Dropdown(choices=choices, label="FLUX.1-schnell", interactive=True)

    with gr.Tab("Deploy"):
        gr.Markdown("Deploy Models TODO")


with demo.route("Settings"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)

demo.launch(
    theme=gr.themes.Soft(),
    css="""
    footer {
        visibility: hidden
    } 
    #chatbot-section {
        min-height: 65vh
    } 
    #InputField .record.record-button {
        margin-top: 15%
    } 
    #InputField .audio-container.compact-audio {
        margin-top: 5%
    }
    """,
    favicon_path=os.path.join("src", "apolloai", "images", "owl.png")
)
