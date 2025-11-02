from langchain.agents import initialize_agent, AgentType

def build_agent(llm, tools):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DESCRIPTION,
        verbose=True
    )
    return agent
