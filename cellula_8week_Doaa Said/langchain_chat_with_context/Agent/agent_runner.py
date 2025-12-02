from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

def build_agent(llm, tools):
    """
    Builds and returns an agent using the v1.x `create_agent` API.
    The returned object is already an executable agent (it handles tool execution).
    """

    # System prompt for the agent
    system_prompt = """You are a helpful AI assistant with access to tools.
    Your job is to help users answer questions by using the available tools when needed.

    Available tools:
    - ContextPresenceJudge: Checks if enough context or document text is present to answer the question
    - ContextRelevanceChecker: Determines if the context is relevant to the question
    - ContextSplitter: Splits a user message into 'context' and 'question' parts
    - WebSearchTool: Searches Wikipedia for missing context or information

    Workflow:
    1. First check if the user has provided enough context
    2. If not, use web search
    3. Check if the available context is relevant
    4. Provide helpful, accurate answers

    Always be polite and helpful."""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_agent(
        model=llm,
        tools=tools,
         system_prompt= system_prompt 
        
    )

    return agent