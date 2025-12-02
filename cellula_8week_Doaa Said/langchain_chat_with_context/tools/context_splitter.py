from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

def build_context_splitter_tool(llm, prompt_file_path="prompts/context_splitter_prompt.txt"):
    """
    Creates a tool that splits a user message into 'context' and 'question'.
    Returns a JSON string to ensure LangChain agent compatibility.
    """

    # Load prompt from external file
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    # Create LCEL prompt template
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Build chain
    chain = prompt | llm | StrOutputParser()

    @tool("ContextSplitter")  
    def context_splitter(message: str) -> str:
        """
        Splits a user message into 'context' and 'question'.
        Returns a JSON string.
        """
        raw_output = chain.invoke({"message": message})
        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            result = {"context": "", "question": message}
        return json.dumps(result)  

    return context_splitter
