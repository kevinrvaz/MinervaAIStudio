import json
import os
from copy import deepcopy
from datetime import datetime

import gradio as gr
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from minervaai.tools import *

gpt_oss_system_prompt = f"""
You are Minerva named after the roman goddess, an AI assistant.
Knowledge cutoff: 2024-06
Current date: {str(datetime.now().date())}

General Behavior
- Speak in a friendly, helpful tone.
- Provide clear, concise answers unless the user explicitly requests a more detailed explanation.
- Use the user's phrasing and preferences; adapt style and formality to what the user indicates.
- If a user asks for a change (e.g., a different format or a deeper dive), obey unless it conflicts with policy or safety constraints.

Reasoning Depth
- Default reasoning level is “medium”: generate a quick chain of thought then produce the final answer.
- If the user requests a detailed walk-through, raise the reasoning depth (“high”) to produce a step-by-step analysis.

Memory & Context
- Only retain the conversation context within the current session; no persistent memory after the session ends.
- Use up to the model's token limit (≈8k tokens) across prompt + answer. Trim or summarize as needed.

Safety & Filtering
- Apply content policy filters to all outputs. Disallowed content includes but is not limited to: hate speech, self-harm encouragement, disallowed advice, disallowed content about minors, disallowed medical or legal advice, etc.
- If a user request conflicts with policy, refuse, safe-complete, or offer a partial answer subject to the policy.

Response Formatting Options
- Recognize prompts that request specific formats (e.g., Markdown code blocks, bullet lists, tables).
- If no format is specified, default to plain text with line breaks; include code fences for code.

Language Support
- Primarily English by default; can switch to other languages if the user explicitly asks.

Developer Instructions (meta-settings)
- Identity: “You are Minerva named after the roman goddess, an AI assistant.”
- Knowledge cutoff: 2024-06
- Current date: {str(datetime.now().date())}
- Reasoning depth: “medium” by default, updatable via user request.
- Interaction style: friendly, collaborative, concise unless otherwise requested.
- External tool access: enabled.
- Memory: session-only, no long-term retention.
- Output size: keep responses < 800-1,000 words unless specifically requested otherwise.
"""

LLM_CONFIG = {
    "gpt-oss:20b": {
        "tags": ["text-generation", "tool-calling", "thinking"],
        "system_prompt": gpt_oss_system_prompt,
        "inference_parameters": {
            "temperature": {
                "min": 0,
                "max": 1,
                "type": "slider",
                "default": 0.8,
                "order": 1,
                "label": "Temperature",
            },
            "reasoning": {
                "default": "medium",
                "type": "dropdown",
                "options": ["low", "medium", "high"],
                "order": 0,
                "label": "Reasoning",
            },
            "num_ctx": {
                "default": 2048,
                "type": "number",
                "min": 1,
                "max": 128000,
                "order": 2,
                "label": "Context Length",
            },
            "top_k": {
                "default": 40,
                "type": "number",
                "min": 1,
                "max": 100,
                "order": 3,
                "label": "Top K Sampling",
            },
            "top_p": {
                "default": 0.9,
                "type": "slider",
                "min": 0,
                "max": 1,
                "order": 4,
                "label": "Top P Sampling",
            },
            "repeat_penalty": {
                "default": 1.1,
                "type": "number",
                "min": 0,
                "max": 2,
                "order": 5,
                "label": "Repeat Penalty",
            },
            "num_predict": {
                "default": 1000,
                "type": "number",
                "min": -1,
                "max": 128000,
                "order": 6,
                "label": "Max Tokens",
            },
            "seed": {
                "default": None,
                "type": "number",
                "min": None,
                "max": None,
                "label": "Seed",
                "order": 7,
            },
        },
    }
}


def set_parameter_field(parameter, val, config):
    return {**config, parameter: val}


def load_history(file_path):
    with open(file_path) as file:
        data = json.load(file)

    history = data["messages"]
    denormalize_history(history)

    # Todo check other fields reload or not
    return (
        history,
        set(data["builtin_tools_selected"]),
        data["model_config"],
        data["model_selected"],
        data["session_id"],
    )


def clear_history(file_path, render_state, current_session):
    if os.path.split(file_path)[-1] == f"{current_session}.json":
        gr.Warning("Cannot delete current session ⛔️!", duration=5)
        return render_state

    if os.path.exists(file_path):
        os.unlink(file_path)

    return not render_state


def record_history(
    history,
    current_chat_session,
    model_selected,
    model_config_state,
    builtin_tools_selected,
):
    history_dump = deepcopy(history)
    normalize_history(history_dump)
    with open(
        os.path.join("message_histories", f"{current_chat_session}.json"), "w"
    ) as history_file:
        json.dump(
            {
                "messages": history_dump,
                "session_id": current_chat_session,
                "model_selected": model_selected,
                "builtin_tools_selected": list(builtin_tools_selected),
                "model_config": model_config_state,
                "timestamp": str(datetime.now()),
            },
            history_file,
        )


def history_builder(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=True)


def get_agent(model_name, model_parameters, tools_selected):
    print(model_name, model_parameters)
    llm = ChatOllama(model=model_name, **model_parameters)
    agent = create_agent(
        llm,
        tools=filter_tools(tools_selected),
        system_prompt=LLM_CONFIG[model_name]["system_prompt"],
    )
    return agent


def normalize_history(history):
    for message in history:
        if "metadata" in message and message["metadata"] is None:
            del message["metadata"]

        if (
            message["role"] == "user"
            and isinstance(message["content"][0], dict)
            and "file" in message["content"][0]
        ):
            message["metadata"] = {
                "title": "file_contents",
                "log": message["content"][0]["file"],
            }

            message["content"] = f"file path - {message["content"][0]["file"]}"

        if (
            message["role"] == "assistant"
            and "metadata" in message
            and (
                "image" in message["metadata"]["title"]
                or "3d" in message["metadata"]["title"]
                or "speech" in message["metadata"]["title"]
                or "video" in message["metadata"]["title"]
            )
        ):
            message["content"] = message["metadata"]["log"]


def denormalize_history(history):
    for message in history:
        if "metadata" in message and message["metadata"] is None:
            del message["metadata"]

        if (
            message["role"] == "assistant"
            and "metadata" in message
            and (
                "image" in message["metadata"]["title"]
                or "3d" in message["metadata"]["title"]
                or "speech" in message["metadata"]["title"]
                or "video" in message["metadata"]["title"]
            )
        ):
            if os.path.isfile(message["metadata"]["log"]):
                if "3d" in message["metadata"]["title"]:
                    message["content"] = gr.Model3D(
                        message["metadata"]["log"],
                    )
                elif "image" in message["metadata"]["title"]:
                    message["content"] = gr.Image(
                        message["metadata"]["log"],
                        buttons=["download", "share", "fullscreen"],
                    )
                elif "speech" in message["metadata"]["title"]:
                    message["content"] = gr.Audio(
                        message["metadata"]["log"],
                        buttons=["download", "share"],
                    )
                elif "video" in message["metadata"]["title"]:
                    message["content"] = gr.Video(
                        message["metadata"]["log"],
                        buttons=["download", "share"],
                    )

        if (
            message["role"] == "user"
            and "metadata" in message
            and "file_contents" in message["metadata"]["title"]
        ):
            message["content"] = message["metadata"]["log"]
            message["metadata"] = None


def chat_completion(history, model_name, model_parameters, tools_selected):
    normalize_history(history)
    for chunk in get_agent(model_name, model_parameters, tools_selected).stream(
        {"messages": history}, stream_mode="updates"
    ):
        for step, data in chunk.items():
            if step == "model":
                for message in data["messages"][-1].content_blocks:
                    if message["type"] == "reasoning":
                        history.append(
                            {
                                "role": "assistant",
                                "content": message["reasoning"],
                                "metadata": {"title": "💭 Thought", "staus": "done"},
                            }
                        )
                    elif message["type"] == "text":
                        history.append(
                            {"role": "assistant", "content": message["text"]}
                        )
                    elif message["type"] == "tool_call":
                        history.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "metadata": {
                                    "title": f"🛠️ Used {message["name"]} tool",
                                    "staus": "pending",
                                    "log": "In Progress",
                                },
                            }
                        )
                    else:
                        print("Unknown message", message)
            elif step == "tools":
                for message in data["messages"][-1].content_blocks:
                    if message["type"] == "text":
                        history[-1]["content"] = message["text"]
                        history[-1]["metadata"]["status"] = "done"
                        history[-1]["metadata"]["log"] = (
                            message["text"] if os.path.isfile(message["text"]) else ""
                        )
                    else:
                        print("Unknown message", message)
            else:
                print("Unknown step", step, data)

            denormalize_history(history)
            yield history


def failed_chat_completion(history):
    history.append({"role": "assistant", "content": "User terminated execution"})
    return history
