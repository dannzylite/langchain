from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

def search(query: str) -> str:
    """
    This Tool Searches the web for a given query

    Args:
        query: The query to search for
    Returns:
        The search results as a string    
    """

    print(f"Searching for: {query}")
    return tavily.search(query=query)

tools = [search]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model=llm, tools=tools)


def main():
    result =agent.invoke({"messages": [HumanMessage(content="search for 3 job posting for an ai engineeer using langchain in the bay area on linkedin and list their details")]})
    print(result)

if __name__ == "__main__":
    main()