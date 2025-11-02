from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def build_context_presence_tool(llm):
    with open("prompts/context_judge_prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()

    prompt = PromptTemplate.from_template(prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt)

    return Tool.from_function(
        func=chain.run,
        name="ContextPresenceJudge",
        description="Determines whether user input includes background context or is just a direct question."
    )