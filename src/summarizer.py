"""
Document summarization functionality.
"""
from typing import Dict, List, Any, Optional, Union

from .models.claude_interface import ClaudeInterface
from .text_processor import TextProcessor

class Summarizer:
    """Generate summaries of documents at various levels of detail."""
    
    def __init__(self, claude_interface: ClaudeInterface, text_processor: Optional[TextProcessor] = None):
        """Initialize the summarizer.
        
        Args:
            claude_interface: Claude API interface
            text_processor: Text processor for chunking (optional)
        """
        self.claude = claude_interface
        self.text_processor = text_processor or TextProcessor()
    
    def generate_document_summary(self, 
                                  document: Dict[str, Any], 
                                  length: str = "medium",
                                  focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a summary of the entire document.
        
        Args:
            document: Document dictionary from DocumentLoader
            length: Summary length ('very_short', 'short', 'medium', 'long', 'very_long')
            focus_areas: Optional list of areas to focus on
            
        Returns:
            Dictionary with summary and metadata
        """
        # Map length to approximate token count
        length_to_tokens = {
            "very_short": 150,
            "short": 300,
            "medium": 600,
            "long": 1000,
            "very_long": 2000
        }
        
        # Use document text, but limit to a reasonable size
        text = document.get("full_text", "")
        if len(text) > 25000:  # Limit to ~25k chars to stay within token limits
            text = text[:25000]
        
        # Prepare prompt for Claude
        prompt = self._create_summary_prompt(
            text=text,
            document_type=document.get("document_type", "document"),
            length=length,
            focus_areas=focus_areas
        )
        
        # Get response from Claude
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are a professional document summarization assistant. Create clear, accurate summaries that capture the essence of the document.",
            max_tokens=length_to_tokens.get(length, 600)
        )
        
        return {
            "summary": response["content"],
            "length": length,
            "focus_areas": focus_areas,
            "usage": response.get("usage", {}),
            "document_id": document.get("file_path", "")
        }
    
    def generate_section_summaries(self, 
                                  document: Dict[str, Any],
                                  section_headers: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Generate summaries for individual document sections.
        
        Args:
            document: Document dictionary from DocumentLoader
            section_headers: Optional list of section headers (if None, try to detect automatically)
            
        Returns:
            List of section summaries
        """
        text = document.get("full_text", "")
        
        # If no section headers provided, try to detect them
        if not section_headers:
            section_headers = self.text_processor.find_section_headers(text)
        
        if not section_headers:
            return []
        
        # Process sections
        sections = []
        for i, header in enumerate(section_headers):
            # Get section text (from this header to next, or end)
            start_pos = header["char_position"]
            end_pos = len(text)
            if i < len(section_headers) - 1:
                end_pos = section_headers[i+1]["char_position"]
            
            section_text = text[start_pos:end_pos].strip()
            
            # Skip very short sections
            if len(section_text) < 200:
                continue
            
            # Create summary prompt for this section
            prompt = f"""
            Summarize the following section from a document:
            
            SECTION HEADER: {header["text"]}
            
            SECTION TEXT:
            ```
            {section_text[:5000]}  # Limit to ~5k chars
            ```
            
            Provide a concise summary of this section in 2-3 sentences.
            """
            
            # Get response
            response = self.claude.generate_response(
                prompt=prompt,
                system_prompt="You are a document section summarization assistant. Create brief, focused summaries of document sections.",
                max_tokens=200
            )
            
            sections.append({
                "header": header["text"],
                "char_position": header["char_position"],
                "summary": response["content"],
                "usage": response.get("usage", {})
            })
        
        return sections
    
    def generate_executive_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary focused on key takeaways.
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            Dictionary with executive summary and metadata
        """
        text = document.get("full_text", "")
        if len(text) > 25000:
            text = text[:25000]
        
        # Create executive summary prompt
        prompt = f"""
        Create an executive summary of the following document:
        
        ```
        {text}
        ```
        
        Your executive summary should:
        1. Begin with a 1-paragraph overview
        2. List 3-5 key takeaways as bullet points
        3. Include any critical findings or recommendations
        4. Be concise and focused on decision-relevant information
        
        Format the summary for busy executives who need to quickly understand the most important points.
        """
        
        # Get response
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are an executive communication specialist. Create focused, action-oriented executive summaries that highlight decision-relevant information.",
            max_tokens=800
        )
        
        return {
            "executive_summary": response["content"],
            "usage": response.get("usage", {}),
            "document_id": document.get("file_path", "")
        }
    
    def _create_summary_prompt(self, 
                              text: str, 
                              document_type: str = "document",
                              length: str = "medium",
                              focus_areas: Optional[List[str]] = None) -> str:
        """Create a prompt for document summarization.
        
        Args:
            text: Document text to summarize
            document_type: Type of document (for context)
            length: Summary length
            focus_areas: Optional areas to focus on
            
        Returns:
            Formatted prompt string
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
        
        # Create base prompt
        prompt = f"""
        Please provide {length_desc} summary of the following {document_type}.
        
        {focus_instruction}
        
        {document_type.upper()}:
        ```
        {text}
        ```
        
        Provide a summary that captures the main points, key details, and overall message of the {document_type}.
        """
        
        return prompt


def generate_summary(document: Dict[str, Any], 
                    claude_interface: ClaudeInterface, 
                    length: str = "medium",
                    focus_areas: Optional[List[str]] = None) -> str:
    """Convenience function to generate a document summary.
    
    Args:
        document: Document dictionary from DocumentLoader
        claude_interface: Claude API interface
        length: Summary length
        focus_areas: Optional areas to focus on
        
    Returns:
        Summary text
    """
    summarizer = Summarizer(claude_interface=claude_interface)
    result = summarizer.generate_document_summary(
        document=document,
        length=length,
        focus_areas=focus_areas
    )
    return result["summary"]
