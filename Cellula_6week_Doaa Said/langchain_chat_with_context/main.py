from langchain_community.llms import Ollama
from tools.context_presence_judge import build_context_presence_tool
from tools.web_search_tool import WebSearchTool
from tools.context_relevance_checker import build_context_relevance_tool
from tools.context_splitter import build_context_splitter_tool
from Agent.agent_runner import build_agent

# Initialize local model (LLaMA 3 or other)
llm = Ollama(model="llama3")

# Build tools
context_judge = build_context_presence_tool(llm)
relevance_checker = build_context_relevance_tool(llm)
splitter = build_context_splitter_tool(llm)

tools = [context_judge, relevance_checker, splitter, WebSearchTool]

# Build the intelligent agent
agent = build_agent(llm, tools)

# Test run
response = agent.run("Explain the use of transformers in NLP.")
print("\n🤖 Agent Response:\n", response)
