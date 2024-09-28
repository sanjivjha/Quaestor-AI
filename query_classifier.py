"""
This file contains the query classification functionality, including a base class for custom classifiers.
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class QueryClassification(BaseModel):
    strategy: str = Field(description="The RAG strategy to be used")
    explanation: str = Field(description="Brief explanation for the chosen strategy")

class BaseClassifier:
    def predict(self, query: str) -> QueryClassification:
        raise NotImplementedError

class RuleBasedQueryClassifier(BaseClassifier):
    def __init__(self, llm):
        self.llm = llm
        self.classification_chain = self._create_classification_chain()

    def _create_classification_chain(self):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze the following query and determine the most appropriate RAG (Retrieval-Augmented Generation) approach. Consider the following options:

1. "simple_rag": For straightforward, factual queries that likely have a direct answer in the knowledge base.
2. "multi_document_rag": For queries that might require synthesizing information from multiple documents or sections.
3. "time_sensitive_rag": For queries about recent events or time-sensitive information that might require up-to-date sources.
4. "calculation_rag": For queries involving numerical calculations or data analysis.
5. "general_knowledge": For queries that don't require specific document retrieval and can be answered with general knowledge.

Query: {query}

{format_instructions}
"""
        )
        parser = PydanticOutputParser(pydantic_object=QueryClassification)
        return LLMChain(
            llm=self.llm, 
            prompt=prompt.partial(format_instructions=parser.get_format_instructions()),
            output_parser=parser
        )

    def predict(self, query: str) -> QueryClassification:
        return self.classification_chain.predict(query=query)