"""
This file serves as an enhanced test driver for the ResearchPal AI system.
It demonstrates and tests various capabilities of the system, including
document ingestion, Wikipedia integration, knowledge base querying,
and the iterative RAG process.
"""

import os
import requests
import sys
import boto3
import botocore

def check_aws_credentials():
    try:
        session = boto3.Session()
        sts = session.client('sts')
        sts.get_caller_identity()
        
        try:
            bedrock = session.client('bedrock-runtime')
        except botocore.exceptions.UnknownServiceError:
            print("Error: AWS Bedrock is not available in the current region.")
            print("Please ensure you're using a region where Bedrock is available.")
            sys.exit(1)
        
    except botocore.exceptions.NoCredentialsError:
        print("Error: AWS credentials not found.")
        print("Please configure your AWS credentials using 'aws configure'.")
        sys.exit(1)
    except botocore.exceptions.ClientError as e:
        print(f"Error: AWS credentials are invalid or lack necessary permissions.")
        print(f"Error details: {str(e)}")
        sys.exit(1)

def check_dependencies():
    missing_dependencies = []
    try:
        from self_rag_system import SelfRAGSystem
    except ImportError as e:
        missing_dependencies.append(str(e))
    
    if missing_dependencies:
        print("Error: Missing dependencies or modules:")
        for dep in missing_dependencies:
            print(f"- {dep}")
        print("\nPlease ensure all required files and dependencies are installed.")
        print("You may need to run: pip install -r requirements.txt")
        sys.exit(1)

    return SelfRAGSystem

def download_pdf(url: str, filename: str):
    response = requests.get(url)
    if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        return False

def test_document_ingestion(rag_system):
    print("\n--- Testing Document Ingestion ---")
    pdf_filename = "apple_annual_report_2023.pdf"
    
    if os.path.exists(pdf_filename):
        print(f"Ingesting {pdf_filename}...")
        try:
            rag_system.ingest_pdf(pdf_filename)
            print("PDF ingested successfully.")
        except Exception as e:
            print(f"Error ingesting PDF: {str(e)}")
            print("Skipping further tests that depend on document ingestion.")
    else:
        print(f"PDF file {pdf_filename} not found. Skipping ingestion.")

def test_wikipedia_integration(rag_system):
    print("\n--- Testing Wikipedia Integration ---")
    topics = ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]
    for topic in topics:
        print(f"Adding '{topic}' to knowledge base from Wikipedia...")
        result = rag_system.wikipedia_tool.add_to_knowledge_base(topic)
        print(result)

def test_knowledge_base_query(rag_system):
    print("\n--- Testing Knowledge Base Queries ---")
    queries = [
        "What was Apple's total net sales in 2023?",
        "How much did Apple spend on research and development in 2023?",
        "What are some of the risk factors mentioned in Apple's annual report?",
        "How does Apple approach environmental sustainability according to the report?",
        "What were the main products or services contributing to Apple's revenue in 2023?"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            answer, iterations = rag_system.query(query)
            print(f"Final Answer: {answer}")
            print(f"Number of iterations: {len(iterations)}")
        except Exception as e:
            print(f"Error occurred while processing the query: {str(e)}")


def main():
    check_aws_credentials()
    SelfRAGSystem = check_dependencies()
    try:
        rag_system = SelfRAGSystem()
    except Exception as e:
        print(f"Error initializing SelfRAGSystem: {str(e)}")
        sys.exit(1)
    
    test_document_ingestion(rag_system)
    
    rag_system.setup_agent()
    
    test_knowledge_base_query(rag_system)
    
    print("\n--- Knowledge Base Summary ---")
    summary = rag_system.get_knowledge_base_summary()
    print(summary)

if __name__ == "__main__":
    main()