import os
import random
from uuid import uuid4

import gradio as gr

from minervaai.llm import (
    LLM_CONFIG,
    chat_completion,
    failed_chat_completion,
    history_builder,
    load_history,
    clear_history,
    record_history,
    set_parameter_field,
)
from minervaai.tools import *
from minervaai.tools.general_purpose import search_tool
from minervaai.tools.images import (
    generate_image,
    image_resize_to_new_width,
    image_understanding,
    image_editing,
)

# to be fixed when shifting 3d operations to modal
# from minervaai.tools.threed import (
#     generate_3d_mesh_from_image,
#     generate_3d_model_from_image,
# )
from minervaai.tools.audio import text_to_speech, speech_to_text, music_generation
from minervaai.tools.video import text_to_video
from minervaai.utils import app

main_page = "Agent Mode"

if not os.path.exists("message_histories"):
    os.makedirs("message_histories")

if not os.path.exists("generated_assets"):
    os.makedirs("generated_assets")


@app.local_entrypoint()
def main():
    with gr.Blocks(fill_height=True, title="Minerva AI Studio") as demo:
        model_config_state = gr.State({})
        model_selected = gr.State("gpt-oss:20b")
        builtin_tools_selected = gr.State(default_builtin_tool_setter())
        current_chat_session = gr.State(str(uuid4()))
        rerender_message_histories = gr.State(False)

        @gr.render(
            triggers=[current_chat_session.change, demo.load],
            inputs=[current_chat_session],
        )
        def session_markdown(session_id):
            gr.Markdown(f"Current Chat Session ID: {session_id}")

        chatbot = gr.Chatbot(
            avatar_images=(
                os.path.join("src", "minervaai", "images", "user.png"),
                os.path.join("src", "minervaai", "images", "owl.png"),
            ),
            show_label=False,
            elem_id="chatbot-section",
        )

        navbar = gr.Navbar(visible=True, main_page_name=main_page)

        with gr.Sidebar(open=True):
            gr.Markdown(
                """
                # Minerva AI Studio
                Minerva AI is an AI assistant that can help in creative tasks like creating and editing Images, Video, Audio and 3D structures using open weight models.
                ## Chat Sessions
                """
            )
            with gr.Row():
                new_chat = gr.Button("New Chat")
                new_chat.click(
                    lambda: (str(uuid4()), []), outputs=[current_chat_session, chatbot]
                )

                def get_file_explorer(root_dir, glob_pattern):
                    return gr.FileExplorer(
                        label="Session Histories",
                        glob=glob_pattern,
                        root_dir=root_dir,
                        file_count="single",
                        inputs=[rerender_message_histories],
                        render=False,
                    )

                # dummy input is needed to prevent gradio from caching
                @gr.render(
                    triggers=[
                        demo.load,
                        chatbot.change,
                        rerender_message_histories.change,
                    ],
                    inputs=[rerender_message_histories],
                )
                def message_histories_sidebar(m):
                    # Random choice hack to get gradio to reload file explorer
                    message_history_file = get_file_explorer(
                        "message_histories", random.choice(["*", "*.json"])
                    ).render()
                    load_chat = gr.Button("Load Chat")
                    load_chat.click(
                        load_history,
                        inputs=[message_history_file],
                        outputs=[
                            chatbot,
                            builtin_tools_selected,
                            model_config_state,
                            model_selected,
                            current_chat_session,
                        ],
                    )
                    delete_chat = gr.Button(
                        icon=os.path.join("src", "minervaai", "images", "dustbin.png"),
                        value="",
                    )
                    delete_chat.click(
                        clear_history,
                        inputs=[
                            message_history_file,
                            rerender_message_histories,
                            current_chat_session,
                        ],
                        outputs=[rerender_message_histories],
                    )

        @gr.on(
            triggers=[model_selected.change, demo.load],
            inputs=[model_selected],
            outputs=[model_config_state],
        )
        def default_model_config_setter(model_selected):
            default_params = {}
            if model_selected not in LLM_CONFIG:
                return default_params
            for parameter in sorted(
                LLM_CONFIG[model_selected]["inference_parameters"].keys(),
                key=lambda k: LLM_CONFIG[model_selected]["inference_parameters"][k][
                    "order"
                ],
            ):
                default_params[parameter] = LLM_CONFIG[model_selected][
                    "inference_parameters"
                ][parameter]["default"]
            return default_params

        def build_dynamic_model_settings(model_selected):
            blocks = [gr.Markdown(f"## {model_selected} configuration", render=False)]

            if model_selected not in LLM_CONFIG:
                return blocks + [gr.Markdown("# Config not found", render=False)]

            for parameter in sorted(
                LLM_CONFIG[model_selected]["inference_parameters"].keys(),
                key=lambda k: LLM_CONFIG[model_selected]["inference_parameters"][k][
                    "order"
                ],
            ):
                if (
                    LLM_CONFIG[model_selected]["inference_parameters"][parameter][
                        "type"
                    ]
                    == "slider"
                ):
                    blocks.append(
                        gr.Slider(
                            maximum=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["max"],
                            minimum=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["min"],
                            value=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            label=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                            interactive=True,
                            render=False,
                        )
                    )
                    blocks[-1].change(
                        set_parameter_field,
                        inputs=[
                            gr.State(parameter),
                            blocks[-1],
                            model_config_state,
                        ],
                        outputs=[model_config_state],
                    )
                elif (
                    LLM_CONFIG[model_selected]["inference_parameters"][parameter][
                        "type"
                    ]
                    == "dropdown"
                ):
                    blocks.append(
                        gr.Dropdown(
                            render=False,
                            interactive=True,
                            value=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            choices=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["options"],
                            label=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                        )
                    )
                    blocks[-1].select(
                        set_parameter_field,
                        inputs=[
                            gr.State(parameter),
                            blocks[-1],
                            model_config_state,
                        ],
                        outputs=[model_config_state],
                    )
                elif (
                    LLM_CONFIG[model_selected]["inference_parameters"][parameter][
                        "type"
                    ]
                    == "number"
                ):
                    blocks.append(
                        gr.Number(
                            maximum=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["max"],
                            minimum=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["min"],
                            value=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["default"],
                            label=LLM_CONFIG[model_selected]["inference_parameters"][
                                parameter
                            ]["label"],
                            interactive=True,
                            render=False,
                        )
                    )
                    blocks[-1].change(
                        set_parameter_field,
                        inputs=[
                            gr.State(parameter),
                            blocks[-1],
                            model_config_state,
                        ],
                        outputs=[model_config_state],
                    )
                else:
                    print(
                        f"Unsupported parameter config {LLM_CONFIG[model_selected]["inference_parameters"][parameter]}"
                    )
                    break

            return blocks

        @gr.render(triggers=[model_selected.change, demo.load], inputs=[model_selected])
        def get_agent_settings_sidebar(model_selected):
            with gr.Sidebar(position="right", open=False, width=360):
                gr.Markdown(
                    """
                # Agent Settings
                ## Model Settings
                    """
                )
                for block in build_dynamic_model_settings(model_selected):
                    block.render()

                gr.Markdown("## Agent Tooling")
                with gr.Accordion(label="Builtin Tools", open=False):
                    with gr.Accordion(label=GENERAL_TOOLS["label"], open=False):
                        for tool in GENERAL_TOOLS["tools"]:
                            box = gr.Checkbox(
                                value=tool["default"],
                                label=tool["label"],
                                interactive=True,
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
                                value=tool["default"],
                                label=tool["label"],
                                interactive=True,
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
                                value=tool["default"],
                                label=tool["label"],
                                interactive=True,
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
                                value=tool["default"],
                                label=tool["label"],
                                interactive=True,
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

                    # with gr.Accordion(label=THREED_TOOLS["label"], open=False):
                    #     for tool in THREED_TOOLS["tools"]:
                    #         box = gr.Checkbox(
                    #             value=tool["default"], label=tool["label"], interactive=True
                    #         )
                    #         box.input(
                    #             tool_setter,
                    #             inputs=[
                    #                 gr.State(tool["tool_id"]),
                    #                 builtin_tools_selected,
                    #                 box,
                    #             ],
                    #             outputs=[builtin_tools_selected],
                    #         )

                with gr.Accordion(label="MCP Servers", open=False):
                    gr.Markdown("# TODO")

                gr.Markdown("## Agents")
                with gr.Accordion(label="Agents", open=False):
                    pass

        with gr.Row(equal_height=True):
            with gr.Column(scale=8):
                text_field = gr.MultimodalTextbox(
                    sources=["upload", "microphone"],
                    interactive=True,
                    show_label=False,
                    autofocus=True,
                    placeholder="Send a message to Minerva.",
                    stop_btn=True,
                    elem_id="InputField",
                )

                submit_evt = text_field.submit(
                    history_builder,
                    [chatbot, text_field],
                    [chatbot, text_field],
                )
                chat_success = submit_evt.success(
                    chat_completion,
                    inputs=[
                        chatbot,
                        model_selected,
                        model_config_state,
                        builtin_tools_selected,
                    ],
                    outputs=chatbot,
                )
                chat_success.failure(
                    failed_chat_completion, inputs=[chatbot], outputs=[chatbot]
                ).success(
                    record_history,
                    inputs=[
                        chatbot,
                        current_chat_session,
                        model_selected,
                        model_config_state,
                        builtin_tools_selected,
                    ],
                ).success(
                    lambda val: not val,
                    outputs=[rerender_message_histories],
                    inputs=[rerender_message_histories],
                )

                chat_success.success(
                    record_history,
                    inputs=[
                        chatbot,
                        current_chat_session,
                        model_selected,
                        model_config_state,
                        builtin_tools_selected,
                    ],
                ).success(
                    lambda val: not val,
                    outputs=[rerender_message_histories],
                    inputs=[rerender_message_histories],
                )
                text_field.stop(lambda: print("Stopped bot"), cancels=[chat_success])

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
        navbar = gr.Navbar(visible=True, main_page_name=main_page)

    with demo.route("Agent Arena"):
        navbar = gr.Navbar(visible=True, main_page_name=main_page)

    with demo.route("Builtin Tools"):
        navbar = gr.Navbar(visible=True, main_page_name=main_page)
        with gr.Tab("General Purpose"):
            with gr.Tab("Web Search"):
                with gr.Row():
                    with gr.Column():
                        search_query = gr.TextArea(
                            label="Search Query", interactive=True
                        )
                        search_button = gr.Button("Search")
                    with gr.Column():
                        search_results = gr.TextArea(
                            interactive=False, label="Search Results"
                        )
                    search_button.click(
                        search_tool.invoke,
                        inputs=[search_query],
                        outputs=[search_results],
                    )

        with gr.Tab("Image"):
            with gr.Tab("Image Generation"):
                with gr.Row():
                    with gr.Column():
                        image_query = gr.TextArea(
                            label="Image Prompt", interactive=True
                        )
                        image_button = gr.Button("Generate Image")
                    with gr.Column():
                        image_result = gr.Image(
                            interactive=False, label="Generated Image"
                        )
                    image_button.click(
                        generate_image, inputs=[image_query], outputs=[image_result]
                    )

            with gr.Tab("Image Resizer"):
                with gr.Row():
                    with gr.Column():
                        new_width = gr.Number(label="New width", interactive=True)
                        image_to_resize = gr.Image(
                            label="Image to resize", type="filepath"
                        )
                        resizer_button = gr.Button("Resize")
                    with gr.Column():
                        image_result = gr.Image(
                            interactive=False, label="Resized Image"
                        )
                    resizer_button.click(
                        image_resize_to_new_width,
                        inputs=[image_to_resize, new_width],
                        outputs=[image_result],
                    )

            with gr.Tab("Image Understanding"):
                with gr.Row():
                    with gr.Column():
                        question = gr.TextArea(label="Question", interactive=True)
                        image = gr.Image(label="Image to understand", type="filepath")
                        ask_button = gr.Button("Ask")
                    with gr.Column():
                        answer = gr.TextArea(interactive=False, label="Answer")
                    ask_button.click(
                        image_understanding,
                        inputs=[image, question],
                        outputs=[answer],
                    )

            with gr.Tab("Image Editing"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.TextArea(label="Prompt", interactive=True)
                        image = gr.Image(label="Image to edit", type="filepath")
                        edit_button = gr.Button("Edit Image")
                    with gr.Column():
                        image_result = gr.Image(interactive=False, label="Edited Image")
                    edit_button.click(
                        image_editing,
                        inputs=[image, prompt],
                        outputs=[image_result],
                    )

        with gr.Tab("Video"):
            with gr.Tab("Text to Video"):
                with gr.Row():
                    with gr.Column():
                        video_text = gr.TextArea(label="Video Text", interactive=True)
                        video_negative_prompt = gr.TextArea(
                            label="Video Negative Prompt", interactive=True
                        )
                        video_button = gr.Button("Generate Video")
                    with gr.Column():
                        video_result = gr.Video(
                            interactive=False, label="Generated Video"
                        )
                    video_button.click(
                        text_to_video,
                        inputs=[video_text, video_negative_prompt],
                        outputs=[video_result],
                    )

        with gr.Tab("Audio"):
            with gr.Tab("Text to Speech"):
                with gr.Row():
                    with gr.Column():
                        audio_text = gr.TextArea(label="Audio Text", interactive=True)
                        audio_button = gr.Button("Generate Speech")
                    with gr.Column():
                        audio_result = gr.Audio(
                            interactive=False, label="Generated Speech"
                        )
                    audio_button.click(
                        text_to_speech, inputs=[audio_text], outputs=[audio_result]
                    )

            with gr.Tab("Speech to Text"):
                with gr.Row():
                    with gr.Column():
                        audio_recording = gr.Audio(
                            label="Audio to convert to text",
                            interactive=True,
                            sources=["microphone"],
                            recording=False,
                            type="filepath",
                        )
                        audio_button = gr.Button("Generate Text")
                    with gr.Column():
                        transcribed_audio = gr.TextArea(
                            interactive=False, label="Generated Text"
                        )
                    audio_button.click(
                        speech_to_text,
                        inputs=[audio_recording],
                        outputs=[transcribed_audio],
                    )

            with gr.Tab("Music Generation"):
                with gr.Row():
                    with gr.Column():
                        audio_text = gr.TextArea(label="Text Prompt", interactive=True)
                        audio_button = gr.Button("Generate Music")
                    with gr.Column():
                        audio_result = gr.Audio(
                            interactive=False, label="Generated Music"
                        )
                    audio_button.click(
                        music_generation, inputs=[audio_text], outputs=[audio_result]
                    )

        # with gr.Tab("3D"):
        #     with gr.Tab("Image to 3D Mesh"):
        #         with gr.Column():
        #             image_to_3d = gr.Image(
        #                 label="Image to create 3d mesh from", type="filepath"
        #             )
        #             render_button = gr.Button("Generate 3D Mesh")
        #         with gr.Column():
        #             model_3d = gr.Model3D(interactive=False, label="Generated 3D Mesh")
        #         render_button.click(
        #             generate_3d_mesh_from_image, inputs=[image_to_3d], outputs=[model_3d]
        #         )

        #     with gr.Tab("Image to 3D Model"):
        #         with gr.Column():
        #             image_to_3d = gr.Image(
        #                 label="Image to create 3d model from", type="filepath"
        #             )
        #             render_button = gr.Button("Generate 3D Model")
        #         with gr.Column():
        #             model_3d = gr.Model3D(interactive=False, label="Generated 3D Model")
        #         render_button.click(
        #             generate_3d_model_from_image, inputs=[image_to_3d], outputs=[model_3d]
        #         )

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
        favicon_path=os.path.join("src", "minervaai", "images", "owl.png"),
    )


if __name__ == "__main__":
    main()
