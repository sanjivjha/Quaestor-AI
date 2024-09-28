# ResearchPal AI

ResearchPal AI is an advanced Retrieval-Augmented Generation (RAG) system designed to provide intelligent research assistance. It combines the power of large language models with a flexible knowledge base and customizable components to deliver accurate and context-aware responses to user queries.

## Features

- **Dynamic Knowledge Base**: Ingest PDFs, incorporate Wikipedia content, and add manual text inputs to build a comprehensive knowledge base.
- **Intelligent Query Processing**: Utilizes a multi-stage approach including query classification, retrieval, and answer generation.
- **Answer Evaluation**: Implements a sophisticated evaluation system to assess the quality of generated answers.
- **Iterative Refinement**: Enhances queries and regenerates answers to improve response quality.
- **Extensible Architecture**: Easily add custom tools, evaluators, and classifiers to tailor the system to specific domains.
- **Federated Knowledge Structure**: Supports integration with external knowledge sources for scalable and diverse information retrieval.
- **Transparent Processing**: Provides detailed logs and explanations of the system's decision-making process.
- **User-Friendly Interface**: Streamlit-based GUI for easy interaction and knowledge base management.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sanjivjha/researchpal-ai.git
   cd researchpal-ai
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up AWS credentials for Bedrock access (ensure you have the necessary permissions).

## Usage

Run the Streamlit app:
```
streamlit run streamlit_app.py
```

Use the sidebar to:
- Upload PDF documents
- Add Wikipedia content
- Manually input text to the knowledge base
- View and clear the knowledge base

Use the main chat interface to ask questions and interact with the AI assistant.

## System Components

- `self_rag_system.py`: Core RAG system implementation
- `streamlit_app.py`: User interface
- `query_classifier.py`: Query classification functionality
- `answer_evaluator.py`: Answer evaluation system
- `query_enhancer.py`: Query refinement logic
- `wikipedia_tool.py`: Wikipedia integration

## Extending the System

ResearchPal AI is designed to be highly extensible. Here's how you can extend various components:

### 1. Adding Custom Knowledge Sources

Implement a new knowledge source in the `FederatedKnowledgeBase` class:

```python
def query_custom_db(query: str) -> List[Dict]:
    # Implement your custom database query logic here
    pass

knowledge_base.add_external_source("custom_db", query_custom_db)
```

### 2. Creating Custom Tools

Implement a new tool and add it to the system:

```python
class CustomTool:
    def perform_action(self, input: str) -> str:
        # Implement your tool's functionality
        pass

    def get_tool(self) -> Tool:
        return Tool(
            name="CustomTool",
            func=self.perform_action,
            description="Description of what your tool does"
        )

custom_tool = CustomTool()
rag_system.add_tool(custom_tool.get_tool())
```

### 3. Implementing Custom Answer Evaluators

Create a custom evaluator by extending the `BaseEvaluator` class:

```python
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, query: str, answer: str, context: List[Dict[str, Any]]) -> AnswerEvaluation:
        # Implement your custom evaluation logic
        pass

custom_evaluator = CustomEvaluator()
rag_system.set_answer_evaluator(custom_evaluator)
```

### 4. Developing Custom Query Classifiers

Implement a custom classifier by extending the `BaseClassifier` class:

```python
class CustomClassifier(BaseClassifier):
    def predict(self, query: str) -> QueryClassification:
        # Implement your custom classification logic
        pass

custom_classifier = CustomClassifier()
rag_system.set_query_classifier(custom_classifier)
```

### 5. Enhancing Query Processing

Modify the `query` method in `SelfRAGSystem` to add new processing steps or change the existing flow.

### 6. Expanding the User Interface

Extend the Streamlit interface in `streamlit_app.py` to add new features or visualizations.

## Configuration

- Adjust model parameters in the `SelfRAGSystem` initialization.
- Modify evaluation thresholds in the `_is_answer_satisfactory` method.
- Configure chunk sizes and overlap in the `ingest_pdf` method.

## Debugging

Enable debug mode in the Streamlit interface to view detailed information about the system's operations, including:
- Knowledge base size
- Available tools
- LLM model information
- Embedding model details

## Contributing

Contributions to ResearchPal AI are welcome! Please feel free to submit pull requests, create issues, or suggest new features.

## License

 Apache 2.0

## Contact

Sanjiv Kumar Jha
sanjiv_jha@yahoo.com

---

ResearchPal AI: Empowering research with intelligent, context-aware assistance.# researchpal-ai
