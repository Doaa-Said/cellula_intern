
from langchain.agents import create_agent
def build_agent(llm, tools):
    from langchain.agents import initialize_agent
    
    agent =create_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )
    return agent