from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.tools import tool
import wikipedia


# Import your custom tools
from tools.context_presence_judge import build_context_presence_tool
from tools.context_relevance_checker import build_relevance_checker_tool
from tools.context_splitter import build_context_splitter_tool
from tools.web_search_tool import build_web_search_tool
from Agent.agent_runner import build_agent
def main():
    # 1️⃣ Initialize your local LLM (Ollama must be running locally)
    llm = Ollama(model="llama3")

    # 2️⃣ Build tools
    context_judge = build_context_presence_tool(llm)
    relevance_checker = build_relevance_checker_tool(llm)
    splitter = build_context_splitter_tool(llm)

    tools = [
        context_judge,
        relevance_checker,
        splitter,
      build_web_search_tool
    ]

    # 3️⃣ Create the agent using string agent type
    agent = build_agent(llm,tools)

    # 4️⃣ Run a test query
    query = "Explain the use of transformers in NLP."
    response = agent.run(query)

    print("\n🤖 Agent Response:\n", response)

if __name__ == "__main__":
    main()