"""
Prompt templates for document question answering.
"""

def get_basic_qa_template() -> str:
    """Get a basic prompt template for document Q&A.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Please answer the following question based only on the information provided in the document:
    
    QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Provide a direct answer to the question using only information found in the document.
    If the answer cannot be determined from the document, clearly state this.
    When possible, cite specific sections or quotes from the document to support your answer.
    """
    
    return template


def get_factual_qa_template() -> str:
    """Get a prompt template for factual document Q&A.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Answer the following factual question based exclusively on the information provided in the document:
    
    QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Your answer should:
    1. Provide only facts explicitly stated in the document
    2. Quote directly from the document when possible
    3. Include the location (page, paragraph, section) where the information was found
    4. State clearly if the information is not found in the document
    
    Focus on accuracy and precision, avoiding any speculation or information not provided in the document.
    """
    
    return template


def get_multi_document_qa_template() -> str:
    """Get a prompt template for Q&A across multiple documents.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Answer the following question based on information from multiple documents:
    
    QUESTION: {question}
    
    DOCUMENT 1: {document1_title}
    ```
    {document1_text}
    ```
    
    DOCUMENT 2: {document2_title}
    ```
    {document2_text}
    ```
    
    Your answer should:
    1. Synthesize relevant information from all provided documents
    2. Quote directly from the documents when appropriate, indicating which document you're citing
    3. Note any contradictions or differences in information between documents
    4. State clearly if the information is not found in any of the documents
    
    Focus on providing a comprehensive answer that accurately reflects the information in all documents.
    """
    
    return template


def get_analytical_qa_template() -> str:
    """Get a prompt template for analytical questions about a document.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Provide an analytical answer to the following question based on the document:
    
    QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Your answer should:
    1. Analyze the relevant information from the document
    2. Consider multiple perspectives or interpretations when appropriate
    3. Support your analysis with specific evidence from the document
    4. Provide a nuanced, thoughtful response that goes beyond simple facts
    
    Focus on developing a well-reasoned analysis based solely on the content of the document.
    """
    
    return template


def get_comparison_qa_template() -> str:
    """Get a prompt template for comparing aspects of a document.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Compare the following aspects based on the document:
    
    COMPARISON QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Your answer should:
    1. Identify the key elements being compared
    2. Highlight similarities and differences between them
    3. Support your comparison with specific examples from the document
    4. Organize your response in a clear, structured manner
    
    Focus on providing a balanced comparison based solely on information in the document.
    """
    
    return template


def get_evaluation_qa_template() -> str:
    """Get a prompt template for evaluating arguments or claims in a document.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Evaluate the following based on the document:
    
    EVALUATION QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Your evaluation should:
    1. Identify the relevant arguments, claims, or evidence in the document
    2. Assess the strength, validity, or effectiveness of these elements
    3. Consider any counterarguments or limitations presented
    4. Support your evaluation with specific references to the document
    
    Focus on providing a fair and balanced assessment based solely on the content of the document.
    """
    
    return template


def get_followup_qa_template() -> str:
    """Get a prompt template for follow-up questions in document Q&A.
    
    Returns:
        Formatted prompt template
    """
    template = """
    This is a follow-up question to our previous discussion about the document. Please answer based on the document content:
    
    PREVIOUS QUESTION: {previous_question}
    PREVIOUS ANSWER: {previous_answer}
    
    FOLLOW-UP QUESTION: {question}
    
    DOCUMENT:
    ```
    {document_text}
    ```
    
    Your answer should:
    1. Address the specific follow-up question
    2. Build upon the context of the previous question and answer
    3. Provide new or more detailed information from the document
    4. Maintain consistency with your previous answer
    
    Focus on addressing the follow-up question while maintaining the context of our discussion.
    """
    
    return template
