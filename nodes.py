from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode  #to let langgraph know of the tools that we've
from dotenv import load_dotenv
load_dotenv()

from main import llm, tools

SYSTEM_MESSAGE = """
    You are a helpful assistant than can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState)-> MessagesState:
    """Run the agent reasoning node"""
    response = llm.invoke([{"role":"system", "content": SYSTEM_MESSAGE}, *state["messages"]])  #this will return AI message
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)



