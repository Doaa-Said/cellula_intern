from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.tools import tool

def build_context_presence_tool(llm, prompt_file_path="prompts/context_judge_prompt.txt", document_text=None):
    """
    Creates a tool that checks if the user input already contains enough context
    to answer the question. If a document is loaded, that is considered context.
    """

    # Load prompt template from external file
    with open("prompts\context_judge_prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(prompt_text)],
        input_variables=["document", "input"]
    )


    # Define a wrapper function that calls the LLM with the formatted chat prompt
    def judge_with_input(user_query: str):
        return llm(prompt.format(document=document_text or "", input=user_query))

    # Return a LangChain Tool
    return tool.from_function(
        func=judge_with_input,
        name="ContextPresenceJudge",
        description="Checks if the question already has enough context to answer"
    )
