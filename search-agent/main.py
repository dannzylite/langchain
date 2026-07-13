from typing import List

from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema for a source of information"""

    url: str = Field(description="The source of the information")

class AgentResponse(BaseModel):
    """This is the schema for the agent response"""

    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(default_factory=list, description="The sources of the information")





tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job posting for an ai engineeer using langchain in the bay area on linkedin and list their details"
                )
            ]
        }
    )
    print(result['structured_response'].sources)


if __name__ == "__main__":
    main()


# tavily = TavilyClient()


# def search(query: str) -> str:
#     """
#     This Tool Searches the web for a given query

#     Args:
#         query: The query to search for
#     Returns:
#         The search results as a string
#     """

#     print(f"Searching for: {query}")
#     return tavily.search(query=query)
