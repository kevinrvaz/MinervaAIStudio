import json
import os
from copy import deepcopy
from datetime import datetime

import gradio as gr

from minervaai.common import BASE_IMAGE, read_mcp_config

with BASE_IMAGE.imports():
    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    from langchain_mcp_adapters.client import MultiServerMCPClient

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
- Always think in depth then generate a chain of thought to produce the final answer.

Memory & Context
- Only retain the conversation context within the current session; no persistent memory after the session ends.
- Use up to the model's token limit (≈8k tokens) across prompt + answer. Trim or summarize as needed.

Safety & Filtering
- Apply content policy filters to all outputs. Disallowed content includes but is not limited to: hate speech, self-harm encouragement, disallowed advice, disallowed content about minors, disallowed medical or legal advice, etc.
- If a user request conflicts with policy, refuse, safe-complete, or offer a partial answer subject to the policy.
- Refuse anything that is NSFW

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
        "inference_provider_name": {
            "ollama": "gpt-oss:20b",
            "huggingface": "openai/gpt-oss-20b",
        },
        "system_prompt": gpt_oss_system_prompt,
        "inference_parameters": {
            "temperature": {
                "min": 0,
                "max": 1,
                "type": "slider",
                "default": 0.8,
                "order": 1,
                "label": "Temperature",
                "info": "The temperature of the model. Increasing the temperature will make the model answer more creatively.",
            },
            "reasoning": {
                "default": "medium",
                "type": "dropdown",
                "options": ["low", "medium", "high"],
                "order": 0,
                "label": "Reasoning",
                "info": "Enables reasoning with a custom intensity level."
            },
            "num_ctx": {
                "default": 10000,
                "type": "number",
                "min": 1,
                "max": 128000,
                "order": 2,
                "label": "Context Length",
                "info": "Sets the size of the context window used to generate the next token."
            },
            "top_k": {
                "default": 40,
                "type": "number",
                "min": 1,
                "max": 100,
                "order": 3,
                "label": "Top K Sampling",
                "info": "Reduces the probability of generating nonsense. A higher value (e.g. `100`) will give more diverse answers, while a lower value (e.g. `10`) will be more conservative."
            },
            "top_p": {
                "default": 0.9,
                "type": "slider",
                "min": 0,
                "max": 1,
                "order": 4,
                "label": "Top P Sampling",
                "info": "Works together with top-k. A higher value (e.g., `0.95`) will lead to more diverse text, while a lower value (e.g., `0.5`) will generate more focused and conservative text."
            },
            "repeat_penalty": {
                "default": 1.1,
                "type": "number",
                "min": 0,
                "max": 2,
                "order": 5,
                "label": "Repeat Penalty",
                "info": "Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`) will penalize repetitions more strongly, while a lower value (e.g., `0.9`) will be more lenient."
            },
            "num_predict": {
                "default": 1000,
                "type": "number",
                "min": -1,
                "max": 128000,
                "order": 6,
                "label": "Max Tokens",
                "info": "Maximum number of tokens to predict when generating text."
            },
            "seed": {
                "default": None,
                "type": "number",
                "min": None,
                "max": None,
                "label": "Seed",
                "order": 7,
                "info": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."
            },
            "mirostat": {
                "default": 0,
                "type": "dropdown",
                "options": [0, 1, 2],
                "order": 8,
                "label": "Microstat",
                "info": "Enable Mirostat sampling for controlling perplexity."
            },
            "mirostat_eta": {
                "default": 0.1,
                "type": "number",
                "min": 0,
                "max": 2,
                "order": 9,
                "label": "Mirostat Eta",
                "info": "Influences how quickly the algorithm responds to feedback from generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive."
            },
            "mirostat_tau": {
                "default": 5.0,
                "type": "number",
                "min": 0,
                "max": 10,
                "order": 10,
                "label": "Mirostat Tau",
                "info": "Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text."
            },
            "repeat_last_n": {
                "default": 64,
                "type": "number",
                "min": -1,
                "max": 100000,
                "order": 11,
                "label": "Repeat Last N",
                "info": "Sets how far back for the model to look back to prevent repetition.",
                "precision": 0
            }
        },
    },
    "gpt-oss:120b": {
        "tags": ["text-generation", "tool-calling", "thinking"],
        "inference_provider_name": {
            "ollama": "gpt-oss:120b",
            "huggingface": "openai/gpt-oss-120b",
        },
        "system_prompt": gpt_oss_system_prompt,
        "inference_parameters": {
            "temperature": {
                "min": 0,
                "max": 1,
                "type": "slider",
                "default": 0.8,
                "order": 1,
                "label": "Temperature",
                "info": "The temperature of the model. Increasing the temperature will make the model answer more creatively.",
            },
            "reasoning": {
                "default": "medium",
                "type": "dropdown",
                "options": ["low", "medium", "high"],
                "order": 0,
                "label": "Reasoning",
                "info": "Enables reasoning with a custom intensity level."
            },
            "num_ctx": {
                "default": 10000,
                "type": "number",
                "min": 1,
                "max": 128000,
                "order": 2,
                "label": "Context Length",
                "info": "Sets the size of the context window used to generate the next token."
            },
            "top_k": {
                "default": 40,
                "type": "number",
                "min": 1,
                "max": 100,
                "order": 3,
                "label": "Top K Sampling",
                "info": "Reduces the probability of generating nonsense. A higher value (e.g. `100`) will give more diverse answers, while a lower value (e.g. `10`) will be more conservative."
            },
            "top_p": {
                "default": 0.9,
                "type": "slider",
                "min": 0,
                "max": 1,
                "order": 4,
                "label": "Top P Sampling",
                "info": "Works together with top-k. A higher value (e.g., `0.95`) will lead to more diverse text, while a lower value (e.g., `0.5`) will generate more focused and conservative text."
            },
            "repeat_penalty": {
                "default": 1.1,
                "type": "number",
                "min": 0,
                "max": 2,
                "order": 5,
                "label": "Repeat Penalty",
                "info": "Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`) will penalize repetitions more strongly, while a lower value (e.g., `0.9`) will be more lenient."
            },
            "num_predict": {
                "default": 1000,
                "type": "number",
                "min": -1,
                "max": 128000,
                "order": 6,
                "label": "Max Tokens",
                "info": "Maximum number of tokens to predict when generating text."
            },
            "seed": {
                "default": None,
                "type": "number",
                "min": None,
                "max": None,
                "label": "Seed",
                "order": 7,
                "info": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."
            },
        },
    },
}


def set_parameter_field(parameter, val, config):
    return {**config, parameter: val}


def load_history(file_path):
    with open(file_path) as file:
        data = json.load(file)

    history = data["messages"]
    denormalize_history(history)

    # TODO check other fields reload or not
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


async def get_agent(
    model_name,
    model_parameters,
    tools_selected,
    mcp_servers,
    system_prompt,
    inference_provider,
):
    print(model_name, model_parameters, inference_provider)
    match inference_provider:
        case "ollama":
            llm = ChatOllama(model=model_name, **model_parameters)
        case "huggingface":
            endpoint = HuggingFaceEndpoint(
                repo_id=LLM_CONFIG[model_name]["inference_provider_name"][
                    inference_provider
                ],
                task="text-generation",
                max_new_tokens=model_parameters["num_predict"],
                do_sample=False,
                repetition_penalty=model_parameters["repeat_penalty"],
                provider="groq",
                temperature=model_parameters["temperature"],
                seed=model_parameters["seed"],
                top_p=model_parameters["top_p"],
                top_k=model_parameters["top_k"],
                streaming=True,
            )
            llm = ChatHuggingFace(
                llm=endpoint,
                model_kwargs={
                    "extra_body": {
                        "reasoning_effort": model_parameters["reasoning"],
                        "reasoning_format": "parsed",
                    }
                },
            )
        case _:
            raise ValueError(f"Unsupported inference provider {inference_provider}")

    if mcp_servers:
        mcp_client = MultiServerMCPClient(mcp_servers)
        mcp_tools = await mcp_client.get_tools()
    else:
        mcp_tools = []

    agent = create_agent(
        llm,
        tools=filter_tools(tools_selected) + mcp_tools,
        system_prompt=system_prompt,
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
                or "music" in message["metadata"]["title"]
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
                or "music" in message["metadata"]["title"]
            )
        ):
            if os.path.isfile(message["metadata"]["log"]):
                if "3d" in message["metadata"]["title"]:
                    message["content"] = gr.Model3D(
                        message["metadata"]["log"], min_width=300, height=400
                    )
                elif "image" in message["metadata"]["title"]:
                    message["content"] = gr.Image(
                        message["metadata"]["log"],
                        buttons=["download", "share", "fullscreen"],
                    )
                elif (
                    "speech" in message["metadata"]["title"]
                    or "music" in message["metadata"]["title"]
                ):
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


async def chat_completion(history, model_name, model_parameters, tools_selected, inference_provider, system_prompt):
    mcp_servers = read_mcp_config()
    print("Available mcp servers", mcp_servers)
    normalize_history(history)
    agent = await get_agent(model_name, model_parameters, tools_selected, mcp_servers, system_prompt, inference_provider=inference_provider)
    async for chunk in agent.astream({"messages": history}, stream_mode="updates"):
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
