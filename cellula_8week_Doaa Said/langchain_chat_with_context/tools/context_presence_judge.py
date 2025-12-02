from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_context_presence_tool(llm, prompt_file_path="prompts/context_judge_prompt.txt"):

    # Load prompt
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Build LCEL prompt
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Build LCEL chain
    chain = prompt | llm | StrOutputParser()

    @tool("ContextPresenceJudge")  
    def context_presence_judge(input: str, document: str = "") -> str:
        """Checks if enough context or document text is present to answer the question."""
        return chain.invoke({"input": input})

    return context_presence_judge
