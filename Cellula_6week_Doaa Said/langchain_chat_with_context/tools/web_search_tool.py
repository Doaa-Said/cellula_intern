import wikipedia
from langchain.tools import Tool

def web_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "No results found."

WebSearchTool = Tool.from_function(
    func=web_search,
    name="WebSearchTool",
    description="Searches Wikipedia to retrieve missing context"
)
