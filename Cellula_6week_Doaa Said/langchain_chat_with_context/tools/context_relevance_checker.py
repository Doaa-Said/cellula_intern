from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def build_context_relevance_tool(llm):
    with open("prompts/context_relevance_prompt.txt") as f:
        prompt_text = f.read()
    prompt = PromptTemplate.from_template(prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt)
    return Tool.from_function(
        func=chain.run,
        name="ContextRelevanceChecker",
        description="Checks if provided context is relevant to the user question."
    )
