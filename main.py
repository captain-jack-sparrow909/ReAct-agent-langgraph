from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

@tool
def tripple(num):
    """
    param: a number to tripple
    returns: the tripple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), tripple]

llm = ChatOpenAI(model='gpt-5-nano', temperature=0).bind_tools(tools)



def main():
    print("Hello from react-agent-langgraph!")


if __name__ == "__main__":
    main()
