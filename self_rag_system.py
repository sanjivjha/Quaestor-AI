"""
This file contains the main SelfRAGSystem class, which integrates all components
of the RAG system and manages the question-answering process.

Extensibility Points:
1. FederatedKnowledgeBase: Add custom knowledge sources
2. SelfRAGSystem: Add custom tools, evaluators, and classifiers
3. Custom Tool Creation: Implement new tools for specific functionalities
4. Custom Evaluators: Implement domain-specific answer evaluation logic
5. Custom Classifiers: Implement specialized query classification strategies
"""

from typing import List, Tuple, Dict, Any, Callable
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
import boto3
import re

from query_classifier import RuleBasedQueryClassifier, BaseClassifier, QueryClassification
from answer_evaluator import AnswerEvaluator, AnswerEvaluation, BaseEvaluator
from query_enhancer import QueryEnhancer
from wikipedia_tool import WikipediaTool
from calculator_tool import CalculatorTool
from date_time_tool import DateTimeTool

class FederatedKnowledgeBase:
    def __init__(self, embeddings):
        self.local_store = FAISS.from_texts(["Initial empty knowledge base"], embeddings)
        self.external_sources = {}

    def add_external_source(self, name: str, source: Callable):
        self.external_sources[name] = source

    def query(self, query: str, sources: List[str] = ["local"]) -> List[Dict[str, str]]:
        results = []
        if "local" in sources:
            local_results = self.local_store.similarity_search(query)
            results.extend([
                {"source": "local", "content": self._extract_content(doc)}
                for doc in local_results
            ])
        for source in sources:
            if source in self.external_sources:
                external_results = self.external_sources[source](query)
                results.extend([
                    {"source": source, "content": self._extract_content(doc)}
                    for doc in external_results
                ])
        return results

    def _extract_content(self, doc) -> str:
        if isinstance(doc, Document):
            return doc.page_content
        elif isinstance(doc, dict):
            return doc.get('content', doc.get('page_content', 'No content available'))
        elif isinstance(doc, str):
            return doc
        else:
            return str(doc)

    def add_texts(self, texts: List[str]):
        self.local_store.add_texts(texts)

    def federated_query(self, query: str) -> str:
        all_sources = ["local"] + list(self.external_sources.keys())
        results = self.query(query, sources=all_sources)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            source = result['source']
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            formatted_results.append(f"{i}. Source: {source}\n   Content: {content}")

        return "\n\n".join(formatted_results)

    def get_tool(self) -> Tool:
        return Tool(
            name="Federated Knowledge Base",
            func=self.federated_query,
            description="Queries the federated knowledge base including local and external sources. Use this for comprehensive information retrieval."
        )
    
class SelfRAGSystem:
    def __init__(self, llm_model: str = "anthropic.claude-3-sonnet-20240229-v1:0", debug_mode: bool = False):
        bedrock_client = boto3.client('bedrock-runtime')
        self.llm = ChatBedrock(
            model_id=llm_model,
            client=bedrock_client,
            model_kwargs={"temperature": 0.7, "max_tokens": 500}
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.knowledge_base = FederatedKnowledgeBase(self.embeddings)
        self.tools = []
        self.agent_executor = None
        self.query_classifier = RuleBasedQueryClassifier(self.llm)
        self.answer_evaluator = AnswerEvaluator(self.llm)  # Updated to pass only LLM
        self.query_enhancer = QueryEnhancer(self.llm)
        self.wikipedia_tool = WikipediaTool(self.embeddings)
        self.debug_mode = debug_mode

        self._add_default_tools()

    def _add_default_tools(self):
        # Add Federated Knowledge Base tool
        federated_kb_tool = self.knowledge_base.get_tool()
        self.add_tool(federated_kb_tool)

        # Add Wikipedia tool
        wikipedia_tool = self.wikipedia_tool.get_tool()
        self.add_tool(wikipedia_tool)

        # Add DateTime tool
        date_time_tool = DateTimeTool()
        self.add_tool(date_time_tool.get_tool())

        # Add Calculator tool
        calculator_tool = CalculatorTool()
        self.add_tool(calculator_tool.get_tool())

    def ingest_pdf(self, pdf_path: str) -> int:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Extract text content from documents
            texts = []
            for doc in documents:
                if isinstance(doc, Document):
                    texts.append(doc.page_content)
                elif isinstance(doc, dict) and 'page_content' in doc:
                    texts.append(doc['page_content'])
                else:
                    raise ValueError(f"Unexpected document format: {type(doc)}")
            
            if not texts:
                raise ValueError("No valid text content found in the PDF.")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_texts = text_splitter.split_text("\n".join(texts))
            
            self.knowledge_base.add_texts(split_texts)
            return len(split_texts)
        except Exception as e:
            raise Exception(f"Error ingesting PDF: {str(e)}") from e

    def add_tool(self, tool: Tool):
        """
        Extension Point 3: Add Custom Tools
        Developers can create and add custom tools to extend system capabilities.
        
        Example:
        class CustomCalculatorTool:
            def perform_calculation(self, expression: str) -> str:
                # Implementation of calculation logic
                pass

            def get_tool(self) -> Tool:
                return Tool(
                    name="CustomCalculator",
                    func=self.perform_calculation,
                    description="Performs custom calculations."
                )

        custom_calc = CustomCalculatorTool()
        rag_system.add_tool(custom_calc.get_tool())
        """
        self.tools.append(tool)
        if self.agent_executor:
            self.agent_executor.tools.append(tool)

    def set_answer_evaluator(self, evaluator: BaseEvaluator):
        """
        Extension Point 4: Custom Answer Evaluator
        Developers can implement and set custom answer evaluators.
        
        Example:
        class DomainSpecificEvaluator(BaseEvaluator):
            def evaluate(self, query: str, answer: str) -> AnswerEvaluation:
                # Implementation of domain-specific evaluation logic
                pass

        domain_evaluator = DomainSpecificEvaluator()
        rag_system.set_answer_evaluator(domain_evaluator)
        """
        self.answer_evaluator = evaluator

    def set_query_classifier(self, classifier: BaseClassifier):
        """
        Extension Point 5: Custom Query Classifier
        Developers can implement and set custom query classifiers.
        
        Example:
        class IndustrySpecificClassifier(BaseClassifier):
            def predict(self, query: str) -> QueryClassification:
                # Implementation of industry-specific classification logic
                pass

        industry_classifier = IndustrySpecificClassifier()
        rag_system.set_query_classifier(industry_classifier)
        """
        self.query_classifier = classifier

    def setup_agent(self):
        prompt = PromptTemplate.from_template(
            """You are an AI assistant tasked with answering questions based on the provided tools.
            Use the most appropriate tool for each query, considering the RAG strategy provided.
            If the tools don't provide a satisfactory answer, use your general knowledge.

            You have access to the following tools:

            {tools}

            Use the following format EXACTLY, including the newlines:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Remember to ALWAYS include the "Action:" line after a "Thought:" line, even if you're not using a tool.
            If you're not using a tool, you can use "Action: None" followed by "Action Input: None".

            Begin! Remember to always use the exact format specified above.

            Human: {input}
            AI: Certainly! I'll answer your question step by step using the specified format.

            Question: {input}
            Thought: I need to determine the best approach to answer this question.
            {agent_scratchpad}"""
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True
    )
    def query(self, question: str, max_iterations: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.agent_executor:
            raise ValueError("Agent not set up. Please run setup_agent() first.")
        
        original_question = question
        iterations = []
        
        for i in range(max_iterations):
            try:
                classification = self.query_classifier.predict(question)
                response = self.agent_executor.invoke({"input": f"[Strategy: {classification.strategy}] {question}"})
                
                actual_response = response.get('output', str(response))
                
                context = self.knowledge_base.query(question)
                
                evaluation = self.answer_evaluator.evaluate(question, actual_response, context)
                
                iteration_info = {
                    "iteration": i + 1,
                    "strategy": classification.strategy,
                    "explanation": classification.explanation,
                    "answer": actual_response,
                    "evaluation": evaluation
                }
                iterations.append(iteration_info)
                
                if self._is_answer_satisfactory(evaluation, question):
                    return actual_response, iterations
                
                if i < max_iterations - 1:
                    question = self.query_enhancer.enhance_query(original_question, classification.explanation, actual_response, evaluation)
                    iteration_info["enhanced_query"] = question
            except Exception as e:
                error_message = str(e)
                print(f"Error in iteration {i+1}: {error_message}")
                
                if "Invalid Format" in error_message:
                    # If it's a formatting error, we can try to recover by reminding the agent about the correct format
                    reminder = "Remember to always use the correct format: Thought: ... followed by Action: ... and Action Input: ..."
                    question = f"{reminder} {question}"
                    iterations.append({
                        "iteration": i + 1,
                        "error": error_message,
                        "recovery_attempt": "Reminded agent about correct format"
                    })
                else:
                    iterations.append({
                        "iteration": i + 1,
                        "error": error_message
                    })
                
                if i == max_iterations - 1:
                    return f"Error occurred during query processing: {error_message}", iterations

        return "Unable to generate a satisfactory answer after multiple attempts.", iterations

    def _is_answer_satisfactory(self, evaluation: AnswerEvaluation, question: str, 
                            threshold: float = 0.7, hallucination_threshold: float = 0.3, 
                            date_accuracy_threshold: float = 0.9) -> bool:
        # For date queries or other simple factual queries, be more lenient
        if re.search(r'\b(date|today|current day|time|now)\b', question.lower()):
            return evaluation.accuracy_score >= date_accuracy_threshold

        # Weighted average score
        weights = {
            'relevance': 0.3,
            'completeness': 0.2,
            'accuracy': 0.3,
            'coherence': 0.2
        }
        
        weighted_score = (
            weights['relevance'] * evaluation.relevance_score +
            weights['completeness'] * evaluation.completeness_score +
            weights['accuracy'] * evaluation.accuracy_score +
            weights['coherence'] * evaluation.coherence_score
        )

        # Check if any individual score is critically low
        critical_threshold = 0.4
        if any(getattr(evaluation, f"{aspect}_score") < critical_threshold 
            for aspect in ['relevance', 'accuracy']):
            return False

        return (weighted_score >= threshold and 
                evaluation.hallucination_score <= hallucination_threshold)

    def get_knowledge_base_summary(self) -> str:
        return f"The knowledge base currently contains {self.knowledge_base.local_store.index.ntotal} entries."

    def add_to_knowledge_base(self, text: str) -> int:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        self.knowledge_base.add_texts(texts)
        return len(texts)

    def clear_knowledge_base(self):
        self.knowledge_base = FederatedKnowledgeBase(self.embeddings)
        self.setup_agent()

    def debug_info(self) -> Dict[str, Any]:
        if not self.debug_mode:
            return {}
        
        return {
            "knowledge_base_size": self.knowledge_base.local_store.index.ntotal,
            "tools": [tool.name for tool in self.tools],
            "llm_model": self.llm.model_id,
            "embedding_model": str(self.embeddings),
        }

# Example Extensions -- Instruction how this class can be extended to add more to this. 

# 1. Custom Knowledge Source
def query_enterprise_db(query: str) -> List[Dict]:
    """
    Example of a custom knowledge source that queries an enterprise database.
    
    Args:
        query (str): The search query.
    
    Returns:
        List[Dict]: A list of relevant documents or data points.
    """
    # This is a dummy implementation. In a real scenario, you would connect to your database and perform a query.
    return [
        {"content": f"Enterprise data related to: {query}", "source": "enterprise_db"},
        {"content": f"Additional information for: {query}", "source": "enterprise_db"}
    ]

# Usage:
# knowledge_base.add_external_source("enterprise_db", query_enterprise_db)

# 2. Custom Tool
class CustomCalculatorTool:
    """Example of a custom tool that performs calculations."""
    
    def perform_calculation(self, expression: str) -> str:
        """
        Performs a calculation based on the given expression.
        
        Args:
            expression (str): A mathematical expression as a string.
        
        Returns:
            str: The result of the calculation.
        """
        try:
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    def get_tool(self) -> Tool:
        return Tool(
            name="CustomCalculator",
            func=self.perform_calculation,
            description="Performs custom calculations. Input should be a mathematical expression as a string."
        )

# Usage:
# custom_calc = CustomCalculatorTool()
# rag_system.add_tool(custom_calc.get_tool())

# 3. Custom Answer Evaluator
class SentimentBasedEvaluator(BaseEvaluator):
    """Example of a custom evaluator that considers sentiment in its evaluation."""
    
    def evaluate(self, query: str, answer: str) -> AnswerEvaluation:
        # This is a dummy implementation. In a real scenario, you would use a sentiment analysis model.
        sentiment_score = len(answer) % 5 / 4  # Dummy sentiment score between 0 and 1
        
        return AnswerEvaluation(
            relevance_score=0.8,  # Dummy score
            completeness_score=0.7,  # Dummy score
            accuracy_score=0.9,  # Dummy score
            explanation=f"Evaluation considering sentiment. Sentiment score: {sentiment_score}"
        )

# Usage:
# sentiment_evaluator = SentimentBasedEvaluator()
# rag_system.set_answer_evaluator(sentiment_evaluator)

# 4. Custom Query Classifier
class DomainSpecificClassifier(BaseClassifier):
    """Example of a custom classifier that categorizes queries into specific domains."""
    
    def predict(self, query: str) -> QueryClassification:
        # This is a dummy implementation. In a real scenario, you would use a more sophisticated classification method.
        if "medical" in query.lower():
            strategy = "medical_rag"
        elif "legal" in query.lower():
            strategy = "legal_rag"
        else:
            strategy = "general_rag"
        
        return QueryClassification(
            strategy=strategy,
            explanation=f"Classified as {strategy} based on keyword matching."
        )

# Usage:
# domain_classifier = DomainSpecificClassifier()
# rag_system.set_query_classifier(domain_classifier)

# Example of how to use these extensions
def example_usage():
    rag_system = SelfRAGSystem()
    
    # Add custom knowledge source
    rag_system.knowledge_base.add_external_source("enterprise_db", query_enterprise_db)
    
    # Add custom tool
    custom_calc = CustomCalculatorTool()
    rag_system.add_tool(custom_calc.get_tool())
    
    # Set custom evaluator
    sentiment_evaluator = SentimentBasedEvaluator()
    rag_system.set_answer_evaluator(sentiment_evaluator)
    
    # Set custom classifier
    domain_classifier = DomainSpecificClassifier()
    rag_system.set_query_classifier(domain_classifier)
    
    # Now the system is set up with custom components
    rag_system.setup_agent()
    
    # Example query using the customized system
    query = "What are the legal implications of medical malpractice?"
    answer, iterations = rag_system.query(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"Iterations: {len(iterations)}")

if __name__ == "__main__":
    example_usage()