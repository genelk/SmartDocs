# src/app.py
import streamlit as st
import os
import tempfile
from typing import Dict, List, Any

from document_loader import DocumentLoader
from models.claude_interface import ClaudeInterface
from models.huggingface_models import HuggingFaceManager
from extraction import extract_structured_data
from summarizer import generate_summary

# Initialize components
@st.cache_resource
def load_models():
    claude = ClaudeInterface()
    hf_manager = HuggingFaceManager()
    return claude, hf_manager

claude, hf_manager = load_models()
doc_loader = DocumentLoader(chunk_size=4000, chunk_overlap=200)

# App title and description
st.title("SmartDocs")
st.subheader("Document Analysis and Summary Tool")
st.write("Upload documents for AI-powered analysis, summarization, and information extraction.")

# Document upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md"])

if uploaded_file:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Process document
    with st.spinner("Processing document..."):
        document = doc_loader.load_document(temp_path)
    
    # Display document info
    st.write(f"Document: {uploaded_file.name}")
    st.write(f"Type: {document['document_type']}")
    
    if 'metadata' in document:
        st.write("Metadata:")
        for key, value in document['metadata'].items():
            st.write(f"- {key}: {value}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Information Extraction", "Q&A", "Entity Analysis"])
    
    with tab1:
        st.header("Document Summary")
        
        summary_length = st.select_slider(
            "Summary Length", 
            options=["Very Short", "Short", "Medium", "Long", "Very Long"],
            value="Medium"
        )
        
        focus_area = st.multiselect(
            "Focus Areas (Optional)",
            ["Key Findings", "Methodology", "Results", "Recommendations", "Background"],
            default=[]
        )
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                # Prepare a prompt template for Claude
                prompt = f"""
                Please provide a {summary_length.lower()} summary of the following document. 
                
                {f"Focus on the following areas: {', '.join(focus_area)}" if focus_area else ""}
                
                DOCUMENT:
                ```
                {document['full_text'][:10000]}  # Using first 10k chars for summary
                ```
                
                Provide a concise summary that captures the main points and key details.
                """
                
                response = claude.generate_response(
                    prompt=prompt,
                    system_prompt="You are a professional document summarization assistant. Create clear, accurate summaries that capture the essence of the document.",
                    max_tokens=1000
                )
                
                st.write(response["content"])
                st.write(f"Token usage: {response['usage']['input_tokens']} input, {response['usage']['output_tokens']} output")
    
    with tab2:
        st.header("Information Extraction")
        
        extraction_type = st.selectbox(
            "What information would you like to extract?",
            ["Key People and Organizations", "Dates and Timeline", "Statistical Data", "Custom Extraction"]
        )
        
        if extraction_type == "Custom Extraction":
            custom_extraction = st.text_area("Describe what information to extract:")
        
        if st.button("Extract Information"):
            with st.spinner("Extracting information..."):
                if extraction_type == "Custom Extraction":
                    extraction_template = f"Extract the following information: {custom_extraction}"
                elif extraction_type == "Key People and Organizations":
                    extraction_template = """
                    Extract all people and organizations mentioned in the document with the following details:
                    - For people: name, title/role (if mentioned), associated organization (if mentioned)
                    - For organizations: name, type/industry, key activities mentioned
                    """
                elif extraction_type == "Dates and Timeline":
                    extraction_template = """
                    Create a chronological timeline of events mentioned in the document.
                    For each event include:
                    - Date (as specific as available)
                    - Event description
                    - Significance (if apparent)
                    """
                else:  # Statistical Data
                    extraction_template = """
                    Extract all numerical data and statistics from the document, including:
                    - Figures and percentages
                    - Measurement values
                    - Statistical claims
                    For each, include the context in which it appears.
                    """
                
                # Use Claude for extraction
                result = claude.extract_structured_data(
                    text=document['full_text'][:15000],  # Using first 15k chars
                    extraction_template=extraction_template
                )
                
                st.json(result["data"] if isinstance(result["data"], dict) else {"result": result["data"]})
    
    with tab3:
        st.header("Ask Questions About Document")
        
        question = st.text_input("Enter your question about the document:")
        
        if st.button("Ask") and question:
            with st.spinner("Generating answer..."):
                prompt = f"""
                Please answer the following question based only on the information provided in the document:
                
                QUESTION: {question}
                
                DOCUMENT:
                ```
                {document['full_text'][:20000]}  # Using first 20k chars
                ```
                
                Provide a direct answer to the question using only information found in the document.
                If the answer cannot be determined from the document, clearly state this.
                """
                
                response = claude.generate_response(
                    prompt=prompt,
                    system_prompt="You are a document analysis assistant. Answer questions directly based ONLY on the information in the document. Cite specific sections when possible.",
                    max_tokens=1000
                )
                
                st.write(response["content"])
    
    with tab4:
        st.header("Entity Recognition")
        
        if st.button("Analyze Entities"):
            with st.spinner("Analyzing entities in document..."):
                # Use Hugging Face for NER
                entities = hf_manager.extract_entities(document['full_text'][:10000])
                
                # Display entities by type
                for entity_type, entities_list in entities.items():
                    st.subheader(f"{entity_type}s")
                    unique_entities = {}
                    for entity in entities_list:
                        word = entity["word"]
                        if word not in unique_entities:
                            unique_entities[word] = entity["score"]
                    
                    # Sort by score
                    sorted_entities = sorted(unique_entities.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display top entities
                    for entity, score in sorted_entities[:10]:  # Top 10
                        st.write(f"- {entity} (confidence: {score:.2f})")
    
    # Clean up temp file
    os.unlink(temp_path)
