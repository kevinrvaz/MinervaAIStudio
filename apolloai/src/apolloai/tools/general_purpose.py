from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(output_format="json")

GENERAL_TOOLS = {
    "label": "General Tools",
    "tools": [{"tool": search_tool, "label": "Web Search Tool", "default": True}],
}
