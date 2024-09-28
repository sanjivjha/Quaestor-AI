"""
This file contains the WebSearchTool class, which implements a web search functionality
using the DuckDuckGo search engine.
"""

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool

class WebSearchTool:
    def __init__(self):
        self.search = DuckDuckGoSearchAPIWrapper()

    def search_web(self, query: str) -> str:
        return self.search.run(query)

    def get_tool(self) -> Tool:
        return Tool(
            name="Web Search",
            func=self.search_web,
            description="Useful for finding current information from the web."
        )