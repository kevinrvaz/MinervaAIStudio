from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm_config = {
    "gpt-oss:20b": {
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
            "label": "Top K Sampling"
        },
        "top_p": {
            "default": 0.9,
            "type": "slider",
            "min": 0,
            "max": 1,
            "order": 4,
            "label": "Top P Sampling"
        },
        "repeat_penalty": {
            "default": 1.1,
            "type": "number",
            "min": 0,
            "max": 2,
            "order": 5,
            "label": "Repeat Penalty"
        },
        "num_predict": {
            "default": 128,
            "type": "number",
            "min": -1,
            "max": 128000,
            "order": 6,
            "label": "Max Tokens"
        }
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
- Apply OpenAI’s content policy filters to all outputs. Disallowed content includes but is not limited to: hate speech, self‑harm encouragement, disallowed advice, disallowed content about minors, disallowed medical or legal advice, etc.
- If a user request conflicts with policy, refuse, safe‑complete, or offer a partial answer subject to the policy.
- No external browsing or real‑time data lookup is enabled in this session.

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
- External tool access: disabled (no browsing, no direct API calls).
- Memory: session‑only, no long‑term retention.
- Output size: keep responses < 800–1,000 words unless specifically requested otherwise.
"""

system_message = [{"role": "system", "content": general_prompt}]


def get_agent(model_name, model_parameters):
    print(model_name, model_parameters)
    llm = ChatOllama(model=model_name, **model_parameters)
    agent = create_agent(llm)
    return agent


def completion(history, model_name, model_parameters):
    start_message = system_message
    bot = get_agent(model_name, model_parameters).invoke(
        {"messages": start_message + history}
    )
    bot_message = bot["messages"][-1].content
    if bot["messages"][-1].additional_kwargs:
        history.append(
            {
                "role": "assistant",
                "content": bot["messages"][-1].additional_kwargs["reasoning_content"],
                "metadata": {"title": "Thought", "staus": "done"},
            }
        )

    history.append({"role": "assistant", "content": bot_message})
    return history
