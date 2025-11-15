import gradio as gr
from langchain_community.llms import Ollama

# Import your custom tools
from tools.context_presence_judge import build_context_presence_tool
from tools.context_relevance_checker import build_relevance_checker_tool
from tools.context_splitter import build_context_splitter_tool
from tools.web_search_tool import build_web_search_tool
from Agent.agent_runner import build_agent

def main():
    # 1️⃣ Initialize local LLM
    llm = Ollama(model="llama3")

    # 2️⃣ Build tools (we’ll pass context dynamically later)
    relevance_checker = build_relevance_checker_tool(llm)
    splitter = build_context_splitter_tool(llm)
    web_search = build_web_search_tool()

    tools = [
        relevance_checker,
        splitter,
        web_search
    ]

    # 3️⃣ Build the agent
    agent = build_agent(llm, tools)

    # 4️⃣ Define Gradio interface function
    def run_agent(question: str, user_context: str) -> str:
        # Build a temporary ContextPresenceJudge with user-provided context
        context_judge = build_context_presence_tool(llm, document_text=user_context)

        # Update agent tools dynamically
        agent.tools = [context_judge, relevance_checker, splitter, web_search]

        # Run the agent
        return agent.run(question)

    # 5️⃣ Launch Gradio app
    iface = gr.Interface(
        fn=run_agent,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your question here..."),
            gr.Textbox(lines=5, placeholder="Enter your context here...")
        ],
        outputs=gr.Textbox(label="Agent Response"),
        title="LangChain NLP Agent with User Context",
        description="Ask the agent anything and provide optional context to guide the answer."
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
