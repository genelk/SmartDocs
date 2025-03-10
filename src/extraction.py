"""
Extract structured information from documents.
"""
import re
import json
from typing import Dict, List, Any, Optional, Union

from .models.claude_interface import ClaudeInterface
from .text_processor import TextProcessor

class Extractor:
    """Extract structured information from document content."""
    
    def __init__(self, claude_interface: ClaudeInterface, text_processor: Optional[TextProcessor] = None):
        """Initialize the extractor.
        
        Args:
            claude_interface: Claude API interface
            text_processor: Text processor for chunking (optional)
        """
        self.claude = claude_interface
        self.text_processor = text_processor or TextProcessor()
    
    def extract_entities(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract people, organizations, and locations from document.
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            Dictionary with extracted entities
        """
        text = document.get("full_text", "")
        if len(text) > 15000:
            text = text[:15000]
        
        # Prepare prompt for entity extraction
        prompt = f"""
        Extract all named entities from the following document:
        
        ```
        {text}
        ```
        
        Please identify and categorize the following types of entities:
        1. People (including full names and titles/roles where available)
        2. Organizations (companies, institutions, agencies, etc.)
        3. Locations (cities, countries, addresses, etc.)
        4. Dates and times
        
        For each entity, include a brief description or context if available in the document.
        
        Format the output as valid JSON with the following structure:
        {{
            "people": [
                {{"name": "Person name", "title": "Title/role (if any)", "context": "Brief context"}}
            ],
            "organizations": [
                {{"name": "Organization name", "type": "Type of organization (if known)", "context": "Brief context"}}
            ],
            "locations": [
                {{"name": "Location name", "type": "Type of location", "context": "Brief context"}}
            ],
            "dates": [
                {{"date": "Date or time mentioned", "context": "Brief context"}}
            ]
        }}
        """
        
        # Get response
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are a precise entity extraction assistant. Extract named entities from text and return them in a structured JSON format.",
            max_tokens=2000
        )
        
        # Try to parse JSON from response
        entities = self._extract_json_from_text(response["content"])
        
        # Add metadata
        return {
            "entities": entities,
            "document_id": document.get("file_path", ""),
            "usage": response.get("usage", {})
        }
    
    def extract_data_points(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical data and statistics from document.
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            Dictionary with extracted data points
        """
        text = document.get("full_text", "")
        if len(text) > 15000:
            text = text[:15000]
        
        # Prepare prompt for data point extraction
        prompt = f"""
        Extract all numerical data points and statistics from the following document:
        
        ```
        {text}
        ```
        
        Focus on extracting:
        1. Monetary values (costs, prices, budgets, revenues, etc.)
        2. Percentages and ratios
        3. Quantities and measurements
        4. Dates and time periods
        5. Statistical figures and metrics
        
        For each data point, include:
        - The exact value as mentioned in the text
        - The category or type of data
        - The context (what the number refers to)
        - The unit (if applicable)
        
        Format the output as valid JSON with the following structure:
        {{
            "monetary_values": [
                {{"value": "Value as text", "amount": number, "currency": "Currency code", "context": "Brief context"}}
            ],
            "percentages": [
                {{"value": "Value as text", "percentage": number, "context": "Brief context"}}
            ],
            "quantities": [
                {{"value": "Value as text", "quantity": number, "unit": "Unit of measurement", "context": "Brief context"}}
            ],
            "dates": [
                {{"value": "Date as text", "context": "Brief context"}}
            ],
            "statistics": [
                {{"value": "Value as text", "metric": "Name of metric", "context": "Brief context"}}
            ]
        }}
        
        If you're unsure about a value, include it and note your uncertainty in the context.
        """
        
        # Get response
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are a precise data extraction assistant. Extract numerical data points from text and return them in a structured JSON format.",
            max_tokens=2000
        )
        
        # Try to parse JSON from response
        data_points = self._extract_json_from_text(response["content"])
        
        # Add metadata
        return {
            "data_points": data_points,
            "document_id": document.get("file_path", ""),
            "usage": response.get("usage", {})
        }
    
    def extract_key_points(self, document: Dict[str, Any], num_points: int = 5) -> Dict[str, Any]:
        """Extract key points or takeaways from document.
        
        Args:
            document: Document dictionary from DocumentLoader
            num_points: Number of key points to extract
            
        Returns:
            Dictionary with extracted key points
        """
        text = document.get("full_text", "")
        if len(text) > 20000:
            text = text[:20000]
        
        # Prepare prompt for key point extraction
        prompt = f"""
        Extract the {num_points} most important points or takeaways from the following document:
        
        ```
        {text}
        ```
        
        For each key point:
        1. Provide a concise 1-sentence summary of the point
        2. Include a brief explanation or supporting detail
        3. Note where in the document this point is discussed (if possible)
        
        Format the output as valid JSON with the following structure:
        {{
            "key_points": [
                {{
                    "point": "Concise statement of key point",
                    "explanation": "Brief explanation with additional context or detail",
                    "location": "Section or area of document where discussed (optional)"
                }}
            ]
        }}
        
        Focus on the most significant, substantive information rather than minor details.
        """
        
        # Get response
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are a document analysis assistant. Extract the most important takeaways and return them in a structured format.",
            max_tokens=1500
        )
        
        # Try to parse JSON from response
        key_points = self._extract_json_from_text(response["content"])
        
        # Add metadata
        return {
            "key_points": key_points.get("key_points", []),
            "document_id": document.get("file_path", ""),
            "usage": response.get("usage", {})
        }
    
    def extract_specific_information(self, 
                                   document: Dict[str, Any], 
                                   information_type: str,
                                   custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """Extract specific type of information based on user requirements.
        
        Args:
            document: Document dictionary from DocumentLoader
            information_type: Type of information to extract
            custom_instructions: Optional custom instructions for extraction
            
        Returns:
            Dictionary with extracted information
        """
        text = document.get("full_text", "")
        if len(text) > 15000:
            text = text[:15000]
        
        # Build the extraction prompt based on information type
        prompt_instructions = ""
        if information_type == "contract_terms":
            prompt_instructions = """
            Extract the key contract terms and clauses, including:
            1. Parties involved
            2. Effective dates and duration
            3. Payment terms and amounts
            4. Deliverables and obligations
            5. Termination conditions
            6. Liabilities and warranties
            7. Any special clauses or conditions
            """
        elif information_type == "research_findings":
            prompt_instructions = """
            Extract the key research findings, including:
            1. Study objectives/research questions
            2. Methodology used
            3. Main results and findings
            4. Statistical significance (p-values, confidence intervals, etc.)
            5. Limitations mentioned
            6. Conclusions and implications
            """
        elif information_type == "product_specs":
            prompt_instructions = """
            Extract product specifications, including:
            1. Product name and model
            2. Technical specifications and parameters
            3. Features and capabilities
            4. Dimensions and physical characteristics
            5. Compatibility and requirements
            6. Pricing information (if available)
            """
        else:
            # Use custom instructions or a general fallback
            prompt_instructions = custom_instructions or """
            Extract all relevant information about the document's main topics.
            Include key facts, figures, dates, and any critical information.
            """
        
        # Prepare final prompt
        prompt = f"""
        Extract the following information from this document:
        
        {prompt_instructions}
        
        DOCUMENT:
        ```
        {text}
        ```
        
        Format the output as structured JSON with appropriate fields for each type of information.
        """
        
        # Get response
        response = self.claude.generate_response(
            prompt=prompt,
            system_prompt="You are a precise document extraction assistant. Extract specific information from text and return it in a structured JSON format.",
            max_tokens=2000
        )
        
        # Try to parse JSON from response
        extracted_info = self._extract_json_from_text(response["content"])
        
        # Add metadata
        return {
            "information_type": information_type,
            "extracted_information": extracted_info,
            "document_id": document.get("file_path", ""),
            "usage": response.get("usage", {})
        }
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from text response.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dict (empty dict if parsing fails)
        """
        try:
            # Try to find JSON block with regex
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no match with code blocks, try to parse the whole text
            return json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, try to find anything that looks like JSON
            try:
                # Find the first { and last }
                start = text.find('{')
                end = text.rfind('}')
                
                if start != -1 and end != -1 and end > start:
                    json_str = text[start:end+1]
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Return empty dict if all parsing attempts fail
            return {}


def extract_structured_data(text: str, extraction_template: str, claude_interface: ClaudeInterface, output_format: str = "json") -> Dict[str, Any]:
    """Convenience function to extract structured data from text.
    
    Args:
        text: Text to extract from
        extraction_template: Template guiding extraction
        claude_interface: Claude API interface
        output_format: Format for extracted data
        
    Returns:
        Dictionary with extracted data
    """
    return claude_interface.extract_structured_data(
        text=text,
        extraction_template=extraction_template,
        output_format=output_format
    )
