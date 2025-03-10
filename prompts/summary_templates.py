"""
Prompt templates for document summarization.
"""

def get_document_summary_template(length: str = "medium", focus_areas: list = None) -> str:
    """Get a prompt template for document summarization.
    
    Args:
        length: Summary length ('very_short', 'short', 'medium', 'long', 'very_long')
        focus_areas: Optional list of areas to focus on
        
    Returns:
        Formatted prompt template
    """
    # Map length to descriptive instruction
    length_map = {
        "very_short": "a very brief 1-2 sentence",
        "short": "a concise paragraph-length",
        "medium": "a comprehensive 2-3 paragraph",
        "long": "a detailed multi-paragraph",
        "very_long": "an in-depth, comprehensive"
    }
    
    length_desc = length_map.get(length, "a comprehensive")
    
    focus_instruction = ""
    if focus_areas and len(focus_areas) > 0:
        focus_instruction = f"Focus particularly on the following aspects: {', '.join(focus_areas)}. "
    
    template = f"""
    Please provide {length_desc} summary of the following document.
    
    {focus_instruction}
    
    DOCUMENT:
    ```
    {{document_text}}
    ```
    
    Your summary should:
    1. Capture the main points and key information
    2. Maintain the original meaning and intent
    3. Be concise and well-organized
    4. Exclude minor details and examples unless they're crucial
    
    Provide a summary that accurately represents the document's content and overall message.
    """
    
    return template


def get_executive_summary_template() -> str:
    """Get a prompt template for executive summaries.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Create an executive summary of the following document:
    
    ```
    {document_text}
    ```
    
    Your executive summary should:
    1. Begin with a 1-paragraph overview that captures the document's purpose and main points
    2. List 3-5 key takeaways as bullet points
    3. Include any critical findings, recommendations, or action items
    4. Be concise and focused on decision-relevant information
    5. Avoid technical jargon unless absolutely necessary
    
    Format the summary for busy executives who need to quickly understand the most important points and make decisions based on this information.
    """
    
    return template


def get_section_summary_template() -> str:
    """Get a prompt template for summarizing individual document sections.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Summarize the following section from a document:
    
    SECTION HEADER: {section_header}
    
    SECTION TEXT:
    ```
    {section_text}
    ```
    
    Provide a concise summary of this section in 2-3 sentences. Focus on capturing the main point or purpose of this specific section within the larger document.
    """
    
    return template


def get_comparative_summary_template() -> str:
    """Get a prompt template for comparing multiple documents.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Create a comparative summary of the following documents:
    
    DOCUMENT 1: {document1_title}
    ```
    {document1_text}
    ```
    
    DOCUMENT 2: {document2_title}
    ```
    {document2_text}
    ```
    
    Your comparative summary should:
    1. Provide a brief overview of each document's main purpose and content
    2. Identify key similarities between the documents
    3. Highlight important differences in content, approach, or conclusions
    4. Synthesize the most important information from both documents
    
    Structure your response to clearly distinguish between similarities and differences.
    """
    
    return template


def get_chapter_summary_template() -> str:
    """Get a prompt template for book chapter summarization.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Provide a summary of the following book chapter:
    
    BOOK TITLE: {book_title}
    CHAPTER: {chapter_title}
    
    ```
    {chapter_text}
    ```
    
    Your chapter summary should:
    1. Identify the main themes and purposes of this chapter
    2. Summarize the key events, arguments, or information presented
    3. Explain how this chapter connects to the book's overall narrative or argument
    4. Highlight any significant character development, revelations, or conclusions
    
    Create a comprehensive yet concise summary that would help a reader understand this chapter's content and significance.
    """
    
    return template


def get_research_summary_template() -> str:
    """Get a prompt template for research paper summarization.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Summarize the following research paper:
    
    TITLE: {paper_title}
    
    ```
    {paper_text}
    ```
    
    Your research summary should include:
    1. Research question or objective
    2. Methodology used (data collection, analysis techniques)
    3. Key findings and results
    4. Main conclusions and implications
    5. Limitations mentioned (if any)
    
    Organize your summary using these sections and prioritize clarity and accuracy in describing the research.
    """
    
    return template
