from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain_ollama import ChatOllama
from apolloai.tools.general_purpose import search_tool
from apolloai.tools.images import generate_image_tool, image_resizer_tool
import gradio as gr

llm_config = {
    "gpt-oss:20b": {
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

general_prompt = """
You are Apollo named after the greek god, an AI assistant.
Knowledge cutoff: 2024-06
Current date: 2025-08-06

General Behavior
- Speak in a friendly, helpful tone.
- Provide clear, concise answers unless the user explicitly requests a more detailed explanation.
- Use the user’s phrasing and preferences; adapt style and formality to what the user indicates.
- If a user asks for a change (e.g., a different format or a deeper dive), obey unless it conflicts with policy or safety constraints.

Reasoning Depth
- Default reasoning level is “medium”: generate a quick chain of thought then produce the final answer.
- If the user requests a detailed walk‑through, raise the reasoning depth (“high”) to produce a step‑by‑step analysis.

Memory & Context
- Only retain the conversation context within the current session; no persistent memory after the session ends.
- Use up to the model’s token limit (≈8k tokens) across prompt + answer. Trim or summarize as needed.

Safety & Filtering
- Apply content policy filters to all outputs. Disallowed content includes but is not limited to: hate speech, self‑harm encouragement, disallowed advice, disallowed content about minors, disallowed medical or legal advice, etc.
- If a user request conflicts with policy, refuse, safe‑complete, or offer a partial answer subject to the policy.

Response Formatting Options
- Recognize prompts that request specific formats (e.g., Markdown code blocks, bullet lists, tables).
- If no format is specified, default to plain text with line breaks; include code fences for code.

Language Support
- Primarily English by default; can switch to other languages if the user explicitly asks.

Developer Instructions (meta‑settings)
- Identity: “You are Apollo named after the greek god, an AI assistant”
- Knowledge cutoff: 2024‑06
- Current date: 2025‑08‑06
- Reasoning depth: “medium” by default, updatable via user request.
- Interaction style: friendly, collaborative, concise unless otherwise requested.
- External tool access: enabled.
- Memory: session‑only, no long‑term retention.
- Output size: keep responses < 800–1,000 words unless specifically requested otherwise.
"""


def get_agent(model_name, model_parameters):
    print(model_name, model_parameters)
    llm = ChatOllama(model=model_name, **model_parameters)
    agent = create_agent(
        llm,
        tools=[search_tool, generate_image_tool, image_resizer_tool],
        system_prompt=general_prompt,
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

            message["content"] = f"file path - {message["content"][0]["file"]["path"]}"

        if (
            message["role"] == "assistant"
            and "metadata" in message
            and "image" in message["metadata"]["title"]
        ):
            message["content"] = message["metadata"]["log"]


def denormalize_history(history):
    for message in history:
        if "metadata" in message and message["metadata"] is None:
            del message["metadata"]

        if (
            message["role"] == "assistant"
            and "metadata" in message
            and "image" in message["metadata"]["title"]
        ):
            message["content"] = gr.Image(
                message["metadata"]["log"], buttons=["download", "share", "fullscreen"]
            )

        if (
            message["role"] == "user"
            and "metadata" in message
            and "file_contents" in message["metadata"]["title"]
        ):
            message["content"] = message["metadata"]["log"]
            message["metadata"] = None


def completion(history, model_name, model_parameters):
    normalize_history(history)
    bot = get_agent(model_name, model_parameters).invoke({"messages": history})
    user_message = history[-1]["content"][0]["text"]
    user_index = -1
    for index, message in reversed(list(enumerate(bot["messages"]))):
        try:
            if message.content[0]["text"] == user_message:
                user_index = index
                break
        except Exception as e:
            pass

    for message in bot["messages"][user_index + 1 :]:
        if (
            message.additional_kwargs
            and "reasoning_content" in message.additional_kwargs
        ):
            history.append(
                {
                    "role": "assistant",
                    "content": message.additional_kwargs["reasoning_content"],
                    "metadata": {"title": "💭 Thought", "staus": "done"},
                }
            )

        if isinstance(message, ToolMessage):
            history.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "metadata": {
                        "title": f"🛠️ Used {message.name} tool",
                        "staus": "done",
                        "log": message.content,
                    },
                }
            )
        else:
            history.append({"role": "assistant", "content": message.content})

    denormalize_history(history)
    return history
