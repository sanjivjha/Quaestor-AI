"""
This file contains the QueryEnhancer class, which is responsible for refining
queries based on previous answers and evaluations.
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from answer_evaluator import AnswerEvaluation

class QueryEnhancer:
    def __init__(self, llm):
        self.llm = llm
        self.enhancement_chain = self._create_enhancement_chain()

    def _create_enhancement_chain(self):
        prompt = PromptTemplate(
            input_variables=["original_query", "context", "previous_answer", "evaluation"],
            template="""Given the original query, the context, the previous answer that was deemed unsatisfactory, 
            and its evaluation, please enhance or rephrase the query to get a more relevant, complete, and accurate answer.

            Original Query: {original_query}
            Context: {context}
            Previous Answer: {previous_answer}
            Evaluation: {evaluation}

            Enhanced Query:"""
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def enhance_query(self, original_query: str, context: str, previous_answer: str, evaluation: AnswerEvaluation) -> str:
        return self.enhancement_chain.predict(
            original_query=original_query,
            context=context,
            previous_answer=previous_answer,
            evaluation=evaluation.json()
        )