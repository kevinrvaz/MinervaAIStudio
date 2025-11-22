from apolloai.llm import completion, llm_config
import gradio as gr
from uuid import uuid4

chats = [{"session_id": uuid4(), "messages": [], "short_name": "Dummy Session"}]
main_page = "Agent Mode"


def history_builder(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=True)


with gr.Blocks(
    fill_height=True,
) as demo:
    model_config_state = gr.State({})
    model_selected = gr.State("gpt-oss:20b")

    navbar = gr.Navbar(visible=True, main_page_name=main_page)
    with gr.Sidebar():
        gr.Markdown("# Apollo AI")
        gr.Markdown(
            "Apollo AI is an AI assistant that can help in creative tasks like creating and editing images, video, audio and 3d structures."
        )
        gr.Markdown("## Chat Sessions")
        html_string = ""
        for message in chats:
            html_string += f"<h3>{message['short_name']}</h3>"
        print(html_string)
        gr.Markdown(f" <hr/>{html_string} <hr/>")

    def set_parameter_field(parameter, val, config):
        return {**config, parameter: val}

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
            llm_config[model_selected].keys(),
            key=lambda k: llm_config[model_selected][k]["order"],
        ):
            default_params[parameter] = llm_config[model_selected][parameter]["default"]
        return default_params

    def build_dynamic_model_settings(model_selected):
        blocks = [
            {"field": gr.Markdown(f"## {model_selected} configuration", render=False)}
        ]

        if model_selected not in llm_config:
            return blocks + [{"field": gr.Markdown("# Config not found", render=False)}]

        for parameter in sorted(
            llm_config[model_selected].keys(),
            key=lambda k: llm_config[model_selected][k]["order"],
        ):
            if llm_config[model_selected][parameter]["type"] == "slider":
                blocks.append(
                    {
                        "field": gr.Slider(
                            maximum=llm_config[model_selected][parameter]["max"],
                            minimum=llm_config[model_selected][parameter]["min"],
                            value=llm_config[model_selected][parameter]["default"],
                            label=llm_config[model_selected][parameter]["label"],
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
            elif llm_config[model_selected][parameter]["type"] == "dropdown":
                blocks.append(
                    {
                        "field": gr.Dropdown(
                            render=False,
                            interactive=True,
                            value=llm_config[model_selected][parameter]["default"],
                            choices=llm_config[model_selected][parameter]["options"],
                            label=llm_config[model_selected][parameter]["label"],
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
            elif llm_config[model_selected][parameter]["type"] == "number":
                blocks.append(
                    {
                        "field": gr.Number(
                            maximum=llm_config[model_selected][parameter]["max"],
                            minimum=llm_config[model_selected][parameter]["min"],
                            value=llm_config[model_selected][parameter]["default"],
                            label=llm_config[model_selected][parameter]["label"],
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
                print(f"Unsupported config {llm_config[model_selected][parameter]}")
                break

        return blocks

    @gr.render(triggers=[model_selected.change, demo.load], inputs=[model_selected])
    def get_agent_settings_sidebar(model_selected):
        with gr.Sidebar(position="right", open=False):
            gr.Markdown("# Agent Settings")
            gr.Markdown("## Model Settings")
            for block in build_dynamic_model_settings(model_selected):
                block["field"].render()

            gr.Markdown("## Agent Tooling")
            with gr.Accordion(label="Builtin Tools", open=False):
                gr.Markdown("# TODO")

            with gr.Accordion(label="MCP Servers", open=False):
                gr.Markdown("# TODO")

    chatbot = gr.Chatbot(
        avatar_images=("src/apolloai/images/user.png", "src/apolloai/images/owl.png"),
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
                completion,
                inputs=[chatbot, model_selected, model_config_state],
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

with demo.route("Builtin Tools"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)
    with gr.Tab("Image"):
        pass

    with gr.Tab("Video"):
        pass

    with gr.Tab("Audio"):
        pass

    with gr.Tab("3D"):
        pass

with demo.route("MCP Servers"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)

with demo.route("Model Manager"):
    navbar = gr.Navbar(visible=True, main_page_name=main_page)

    with gr.Tab("Download Models"):
        gr.Markdown("Download Models TODO")

    with gr.Tab("Quantization"):
        gr.Markdown("Quantization Models TODO")

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
)
