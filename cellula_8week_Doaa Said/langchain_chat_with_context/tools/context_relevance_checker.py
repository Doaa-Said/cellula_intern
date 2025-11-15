from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.tools import tool

def build_relevance_checker_tool(llm, prompt_file_path="prompts/relevance_checker_prompt.txt"):
    """
    Creates a tool that checks whether provided context is relevant to a user's question.
    """

    # Load the prompt from an external text file
    with open("prompts\context_relevance_prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Create a ChatPromptTemplate
    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(prompt_text)],
        input_variables=["context", "question"]
    )

    # Define a strong wrapper function for the LLM
    def check_relevance(question: str, context: str) -> str:
        """
        Returns a short response indicating whether the context is relevant to the question.
        Must respond EXACTLY with:
        - 'relevant' if the context answers the question
        - 'irrelevant' if the context does not answer the question
        """
        formatted_prompt = prompt.format(context=context, question=question)
        return llm(formatted_prompt)

    # Return a LangChain Tool
    return tool.from_function(
        func=check_relevance,
        name="ContextRelevanceChecker",
        description="Determines if the provided context is relevant to a specific question. "
                    "Responds with 'relevant' or 'irrelevant' only."
    )
