from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_relevance_checker_tool(llm, prompt_file_path="prompts/context_relevance_prompt.txt"):

    # Load prompt
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Build LCEL prompt
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Build LCEL chain
    chain = prompt | llm | StrOutputParser()

    @tool("ContextRelevanceChecker")  
    def context_relevance_checker(question: str, context: str) -> str:
        """Determines if the context is relevant to the question. (relevant/irrelevant)"""
        return chain.invoke({"question": question, "context": context})

    return context_relevance_checker
