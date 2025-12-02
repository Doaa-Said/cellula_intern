from langchain_community.document_loaders import WikipediaLoader
from langchain.tools import tool

def build_web_search_tool():
    @tool("WebSearchTool")
    def web_search_tool(query: str) -> str:
        """Loads full Wikipedia page content."""
        try:
            docs = WikipediaLoader(query=query, load_max_docs=1).load()
            return docs[0].page_content
        except Exception as e:
            return f"Wikipedia error: {e}"

    return web_search_tool
