import gradio as gr
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
# Import your custom tools
from tools.context_presence_judge import build_context_presence_tool
from tools.context_relevance_checker import build_relevance_checker_tool
from tools.context_splitter import build_context_splitter_tool
from tools.web_search_tool import build_web_search_tool
from Agent.agent_runner import build_agent

def main():
    #  Initialize local LLM
    llm = init_chat_model("ollama:llama3.2")

    #  Build all tools
    print("Building tools...")
    try:
        context_presence_judge = build_context_presence_tool(llm)
        print("✓ Context Presence Judge tool built")
        
        relevance_checker = build_relevance_checker_tool(llm)
        print("✓ Context Relevance Checker tool built")
        
        splitter = build_context_splitter_tool(llm)
        print("✓ Context Splitter tool built")
        
        web_search = build_web_search_tool()
        print("✓ Web Search tool built")
    except Exception as e:
        print(f"Error building tools: {e}")
        return
    
    #  Combine all tools
    tools = [context_presence_judge, relevance_checker, splitter, web_search]
    
    #  Build agent with tools
    print("Building agent...")
    try:
        agent = build_agent(llm, tools)
        print("✓ Agent built successfully")
    except Exception as e:
        print(f"Error building agent: {e}")
        return
    
    #  Create response function for Gradio
    def respond(message, history):
        
        """Process user message through the agent"""
       
            # Invoke the agent with the message
        result = agent.invoke({"messages": message})
        
            
            # Extract the response
           
    
        return result["messages"][-1].content
            
    
    #  Custom CSS for better UI
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .chatbot {
        height: 600px !important;
    }
    """
    
    #  Create Gradio interface
    demo = gr.ChatInterface(
        fn=respond,
        title="🤖 AI Assistant with Context-Aware Tools"

    )
    
    #  Launch the interface
    print("\n🚀 Starting Gradio interface...")
    print("📱 Access the interface at: http://localhost:7860")
    print("Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        
    )

if __name__ == "__main__":
    main()