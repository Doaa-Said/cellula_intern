from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain.tools import tool
import json

def build_context_splitter_tool(llm, prompt_file_path="prompts/splitter_prompt.txt"):
    """
    Creates a tool that splits a user message into 'context' and 'question'.
    """

    # Load prompt from external file
    with open("prompts\context_splitter_prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Create ChatPromptTemplate
    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(prompt_text)],
        input_variables=["message"]
    )
    # Define the splitting function
    def split_message(message: str) -> dict:
        """
        Returns a dict with keys:
        - 'context': extracted context from the message
        - 'question': extracted main question
        """
        formatted_prompt = prompt.format(message=message)
        raw_output = llm(formatted_prompt)
        try:
            data = json.loads(raw_output)
            return data
        except json.JSONDecodeError:
            # Fallback if LLM output is not valid JSON
            return {"context": "", "question": message}

    # Return LangChain Tool
    return tool.from_function(
        func=split_message,
        name="ContextSplitter",
        description="Splits a user message into 'context' and 'question'. Output must be JSON."
    )
