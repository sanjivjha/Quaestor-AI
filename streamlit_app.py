import streamlit as st
import os
import tempfile
from datetime import datetime
import pytz
import base64
from self_rag_system import SelfRAGSystem

class StreamlitInterface:
    def __init__(self):
        self.setup_streamlit()
        self.rag_system = SelfRAGSystem(debug_mode=st.session_state.debug_mode)

    def setup_streamlit(self):
        st.set_page_config(page_title="ResearchPal AI", layout="wide")
        st.title("ResearchPal AI")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "log" not in st.session_state:
            st.session_state.log = []
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = set()

        # Display current date
        current_date = datetime.now(pytz.utc).strftime("%Y-%m-%d")
        st.sidebar.write(f"Current Date: {current_date}")

    def log_action(self, action, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {action}: {details}"
        st.session_state.log.append(log_entry)

    def fallback_file_upload(self):
        st.sidebar.markdown("### Fallback File Upload")
        st.sidebar.markdown("If the regular file upload is not working, you can try this method:")
        file_content = st.sidebar.text_area("Paste the base64 encoded content of your PDF file here:")
        file_name = st.sidebar.text_input("Enter the file name (including .pdf extension):")
        
        if st.sidebar.button("Process Fallback Upload"):
            if file_content and file_name:
                if file_name not in st.session_state.uploaded_files:
                    try:
                        # Decode the base64 content
                        decoded_content = base64.b64decode(file_content)
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(decoded_content)
                            tmp_file_path = tmp_file.name
                        
                        st.sidebar.write(f"Saved to temporary file: {tmp_file_path}")
                        st.sidebar.info(f"Ingesting PDF: {file_name}")
                        
                        added_docs = self.rag_system.ingest_pdf(tmp_file_path)
                        self.log_action("PDF Ingested (Fallback)", f"File: {file_name}, Added docs: {added_docs}")
                        st.sidebar.success(f"PDF ingested successfully! Added {added_docs} documents.")
                        self.rag_system.setup_agent()

                        # Mark this file as uploaded
                        st.session_state.uploaded_files.add(file_name)
                    except Exception as e:
                        st.sidebar.error(f"Error ingesting PDF: {str(e)}")
                        self.log_action("PDF Ingestion Failed (Fallback)", f"File: {file_name}, Error: {str(e)}")
                        st.sidebar.error(f"Detailed error: {repr(e)}")
                    finally:
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)  # Remove the temporary file
                else:
                    st.sidebar.warning(f"File '{file_name}' has already been uploaded and processed.")
            else:
                st.sidebar.error("Please provide both the file content and file name.")

    def run(self):
        st.sidebar.title("ResearchPal AI")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)
        if st.session_state.debug_mode:
            st.sidebar.json(self.rag_system.debug_info())
        
        # Knowledge Base Management
        st.sidebar.header("Knowledge Base Management")
        
        # File uploader with detailed error handling
        uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
        if uploaded_file is not None:
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": uploaded_file.size
            }
            st.sidebar.write("File received by Streamlit:")
            st.sidebar.json(file_details)
            
            if uploaded_file.name not in st.session_state.uploaded_files:
                try:
                    # Read file content
                    file_content = uploaded_file.read()
                    st.sidebar.write(f"Successfully read {len(file_content)} bytes from the file.")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file_content)
                        tmp_file_path = tmp_file.name
                    
                    st.sidebar.write(f"Saved to temporary file: {tmp_file_path}")
                    st.sidebar.info(f"Ingesting PDF: {uploaded_file.name}")
                    
                    added_docs = self.rag_system.ingest_pdf(tmp_file_path)
                    self.log_action("PDF Ingested", f"File: {uploaded_file.name}, Added docs: {added_docs}")
                    st.sidebar.success(f"PDF ingested successfully! Added {added_docs} documents.")
                    self.rag_system.setup_agent()

                    # Mark this file as uploaded
                    st.session_state.uploaded_files.add(uploaded_file.name)
                except Exception as e:
                    st.sidebar.error(f"Error ingesting PDF: {str(e)}")
                    self.log_action("PDF Ingestion Failed", f"File: {uploaded_file.name}, Error: {str(e)}")
                    st.sidebar.error(f"Detailed error: {repr(e)}")
                finally:
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)  # Remove the temporary file
            else:
                st.sidebar.warning(f"File '{uploaded_file.name}' has already been uploaded and processed.")
        
        # Fallback file upload method
        self.fallback_file_upload()
        
        # Text input for manual content addition
        st.sidebar.subheader("Add Content Manually")
        manual_content = st.sidebar.text_area("Enter text to add to the knowledge base:")
        if st.sidebar.button("Add Manual Content"):
            try:
                added_texts = self.rag_system.add_to_knowledge_base(manual_content)
                self.log_action("Manual Content Added", f"Added texts: {added_texts}")
                st.sidebar.success(f"Content added successfully! Added {added_texts} text chunks.")
                self.rag_system.setup_agent()
            except Exception as e:
                st.sidebar.error(f"Error adding manual content: {str(e)}")
                self.log_action("Manual Content Addition Failed", f"Error: {str(e)}")
        
        # Wikipedia search
        st.sidebar.subheader("Add Wikipedia Content")
        wikipedia_query = st.sidebar.text_input("Enter a topic to search on Wikipedia:")
        if st.sidebar.button("Add Wikipedia Content"):
            try:
                result = self.rag_system.wikipedia_tool.search_wikipedia(wikipedia_query)
                added_texts = self.rag_system.add_to_knowledge_base(result)
                self.log_action("Wikipedia Content Added", f"Query: {wikipedia_query}, Added texts: {added_texts}")
                st.sidebar.success(f"Wikipedia content added successfully! Added {added_texts} text chunks.")
                self.rag_system.setup_agent()
            except Exception as e:
                st.sidebar.error(f"Error adding Wikipedia content: {str(e)}")
                self.log_action("Wikipedia Content Addition Failed", f"Query: {wikipedia_query}, Error: {str(e)}")
        
        # Knowledge base summary and clear option
        if st.sidebar.button("Get Knowledge Base Summary"):
            try:
                summary = self.rag_system.get_knowledge_base_summary()
                self.log_action("Knowledge Base Summary Requested")
                st.sidebar.info(summary)
            except Exception as e:
                st.sidebar.error(f"Error getting knowledge base summary: {str(e)}")
                self.log_action("Knowledge Base Summary Failed", f"Error: {str(e)}")
        
        if st.sidebar.button("Clear Knowledge Base"):
            try:
                self.rag_system.clear_knowledge_base()
                self.log_action("Knowledge Base Cleared")
                st.sidebar.success("Knowledge base cleared successfully!")
                st.session_state.uploaded_files.clear()  # Clear the set of uploaded files
            except Exception as e:
                st.sidebar.error(f"Error clearing knowledge base: {str(e)}")
                self.log_action("Knowledge Base Clearing Failed", f"Error: {str(e)}")
        
        # Main chat interface
        st.header("Chat with ResearchPal AI")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Query input
        if query := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = self.process_query(query, message_placeholder)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display log
        with st.expander("View Action Log"):
            for log_entry in st.session_state.log:
                st.text(log_entry)

    def process_query(self, query, message_placeholder):
        self.log_action("Query Received", f"Query: {query}")
        
        try:
            if not self.rag_system.agent_executor:
                self.rag_system.setup_agent()
                self.log_action("Agent Setup", "Agent was not set up, initializing now.")
            
            message_placeholder.markdown("Thinking...")
            response, iterations = self.rag_system.query(query)
            
            full_response = f"**Answer:** {response}\n\n"
            if st.session_state.debug_mode:
                full_response += "**Process Details:**\n"
                for iteration in iterations:
                    full_response += f"Iteration {iteration.get('iteration', 'N/A')}, Retry {iteration.get('retry', 'N/A')}:\n"
                    if 'error' in iteration:
                        full_response += f"- Error: {iteration['error']}\n"
                    else:
                        full_response += f"- Strategy: {iteration.get('strategy', 'N/A')}\n"
                        full_response += f"- Explanation: {iteration.get('explanation', 'N/A')}\n"
                        
                        eval = iteration.get('evaluation')
                        if eval:
                            full_response += f"- Evaluation Scores:\n"
                            full_response += f"  Relevance: {eval.relevance_score:.2f}\n"
                            full_response += f"  Completeness: {eval.completeness_score:.2f}\n"
                            full_response += f"  Accuracy: {eval.accuracy_score:.2f}\n"
                        
                        if 'enhanced_query' in iteration:
                            full_response += f"- Enhanced query: {iteration['enhanced_query']}\n"
                    
                    full_response += "\n"

            self.log_action("Response Generated")
            return full_response
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self.log_action("Query Processing Failed", error_message)
            return error_message

if __name__ == "__main__":
    interface = StreamlitInterface()
    interface.run()