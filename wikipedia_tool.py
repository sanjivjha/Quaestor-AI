"""
This file contains the WikipediaTool class, which provides functionality to search
Wikipedia and add content to the knowledge base.
"""

import wikipedia
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class WikipediaTool:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None

    def search_wikipedia(self, query: str, sentences: int = 5) -> str:
        try:
            return wikipedia.summary(query, sentences=sentences)
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple results found. Please be more specific. Options: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'"

    def add_to_knowledge_base(self, query: str, sentences: int = 5):
        content = self.search_wikipedia(query, sentences)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(content)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
        else:
            self.vector_store.add_texts(texts)
        
        return f"Added Wikipedia content for '{query}' to the knowledge base."

    def get_tool(self) -> Tool:
        return Tool(
            name="Wikipedia Search",
            func=self.search_wikipedia,
            description="Useful for searching Wikipedia for information on a topic. Input should be a search query."
        )

    def get_add_to_kb_tool(self) -> Tool:
        return Tool(
            name="Add to Knowledge Base",
            func=self.add_to_knowledge_base,
            description="Adds Wikipedia content to the knowledge base. Input should be a search query."
        )