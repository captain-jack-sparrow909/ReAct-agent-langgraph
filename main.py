from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


load_dotenv()

#defining some constants:
AGENT_REASON = 'agent_reason'
ACT = 'act'
LAST = -1


# helper functions:
def should_continue(state: MessagesState)->str:
    if not state['messages'][LAST].tool_calls:
        return END
    return ACT

@tool
def triple(num):
    """
    param: a number to triple
    returns: the tripple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model='gpt-5-nano', temperature=0).bind_tools(tools)

SYSTEM_MESSAGE = """
    You are a helpful assistant than can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState)-> MessagesState:
    """Run the agent reasoning node"""
    response = llm.invoke([{"role":"system", "content": SYSTEM_MESSAGE}, *state["messages"]])  #this will return AI message
    return {"messages": [response]}



# defining the flow/graph:
flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)  #The entry point is the node where execution starts.
# Think of it as the graph's "main()" function. Without it, LangGraph wouldn't know which node to run first.
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})
# first arg -> source; 2nd is a function deciding which way to go; 3rd is the mapper

flow.add_edge(ACT, AGENT_REASON)

# app:
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path='flow.png')


def main():
    print("Hello from react-agent-langgraph!")
    query = "what is the temperature in Tokyo now? list it and then triple it"
    res = app.invoke({"messages": [HumanMessage(content=query)]})
    print("---the response is---\n")
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
