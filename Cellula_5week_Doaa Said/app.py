import streamlit as st
from collections import deque

from datasets import load_dataset 
from langgraph.graph import StateGraph, END 
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_ollama import ChatOllama 
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings 
from typing import TypedDict
import re

# ==========================================================
# LLM
# ==========================================================
llm = ChatOllama(
    model="llama3.2",
    temperature=0.6,
    num_ctx=2048,
    base_url="http://localhost:11434",
)


# ==========================================================
# Dataset + Vector Store (RAG)
# ==========================================================
dataset = load_dataset(
    "openai/openai_humaneval",
    split="test"
)
dataset = list(dataset)

documents = []
for item in dataset:
    text = f"Problem:\n{item['prompt']}\n\nSolution:\n{item['canonical_solution']}"
    documents.append(Document(page_content=text))

embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="py_examples",
    embedding_function=embeddings
)

vector_store.add_documents(documents)


def retrieve_code(query: str):
    hits = vector_store.similarity_search(query, k=1)
    return hits[0].page_content if hits else ""


# ==========================================================
# Memory — store the last 5 turns
# ==========================================================
memory_buffer = deque(maxlen=5)


def add_to_memory(role, content):
    memory_buffer.append({"role": role, "content": content})


def generate_memory_prompt():
    if len(memory_buffer) == 0:
        return "No previous memory."
    return "\n".join([f"{m['role']}: {m['content']}" for m in memory_buffer])


# ==========================================================
# LangGraph Shared State
# ==========================================================
class GraphState(TypedDict):
    user_query: str
    chat_state: str
    retrieved_example: str
    final_answer: str
    memory_prompt: str


# ==========================================================
# ROUTER NODE
# ==========================================================
def router(state: GraphState):
    q = state["user_query"].lower()

    if any(k in q for k in ["generate", "write", "create code"]):
        state["chat_state"] = "generate_code"
    elif any(k in q for k in ["explain", "what does", "understand"]):
        state["chat_state"] = "explain_code"
    else:
        state["chat_state"] = "generate_code"

    return state


# ==========================================================
# MEMORY NODE
# ==========================================================
def memory_node(state: GraphState):
    state["memory_prompt"] = generate_memory_prompt()
    return state


# ==========================================================
# RETRIEVER NODE
# ==========================================================
def retrieve_node(state: GraphState):
    state["retrieved_example"] = retrieve_code(state["user_query"])
    return state


# ==========================================================
# GENERATE CODE NODE
# ==========================================================
def generate_code(state: GraphState):
    example = state["retrieved_example"]

    system = f"""
You are a Python code generator. Use RAG example if useful.
Conversation Memory:
{state['memory_prompt']}
"""

    prompt = [
        SystemMessage(content=system),
        HumanMessage(
            content=f"""
User Request:
{state['user_query']}

Relevant Code Example:
{example}

Generate clean and correct Python code only.
"""
        ),
    ]

    result = llm.invoke(prompt)
    state["final_answer"] = result.content
    return state


# ==========================================================
# EXPLAIN CODE NODE
# ==========================================================
def explain_code(state: GraphState):
    example = state["retrieved_example"]

    system = f"""
You are a Python code explainer. Use the retrieved example.
Conversation Memory:
{state['memory_prompt']}
"""

    prompt = [
        SystemMessage(content=system),
        HumanMessage(
            content=f"""
User Request:
{state['user_query']}

Relevant Code Example:
{example}

Explain the code clearly.
"""
        ),
    ]

    result = llm.invoke(prompt)
    state["final_answer"] = result.content
    return state


# ==========================================================
# BUILD LANGGRAPH
# ==========================================================
graph = StateGraph(GraphState)

graph.add_node("router", router)
graph.add_node("memory", memory_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate_code", generate_code)
graph.add_node("explain_code", explain_code)

graph.set_entry_point("router")

graph.add_edge("router", "memory")
graph.add_edge("memory", "retrieve")

graph.add_conditional_edges(
    "retrieve",
    lambda s: s["chat_state"],
    {
        "generate_code": "generate_code",
        "explain_code": "explain_code",
    }
)

graph.add_edge("generate_code", END)
graph.add_edge("explain_code", END)

assistant = graph.compile()


# ==========================================================
# STREAMLIT UI
# ==========================================================

# Set a generic page title and icon
st.set_page_config(page_title="Code Assistant", page_icon="⚙️", layout="wide")

# Display a generic app title
st.set_page_config(page_title="⚡ Code Helper", page_icon="💻", layout="wide")

st.title("💡 Your Coding Companion")

# Chat history visual
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Input box
user_input = st.chat_input("Ask for code or explanation...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    add_to_memory("user", user_input)

    # Run LangGraph
    result = assistant.invoke({
        "user_query": user_input,
        "chat_state": "",
        "retrieved_example": "",
        "final_answer": "",
        "memory_prompt": "",
    })

    output = result["final_answer"]

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": output})
    add_to_memory("assistant", output)

    # Display response
    with st.chat_message("assistant"):
        st.write(output)
#python -m streamlit run
