# src/models/claude_interface.py
import os
import anthropic
from typing import Dict, List, Optional, Any

class ClaudeInterface:
    """Interface for Anthropic's Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize Claude API client.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env variable
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY not found in environment")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate a response from Claude.
        
        Args:
            prompt: The user message
            system_prompt: Optional system prompt to guide Claude's behavior
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing response text and metadata
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                "content": message.content[0].text,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                },
                "model": self.model,
                "id": message.id
            }
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"content": "", "error": str(e)}
            
    def extract_structured_data(self, 
                               text: str, 
                               extraction_template: str,
                               output_format: str = "json") -> Dict[str, Any]:
        """Extract structured data from text using Claude.
        
        Args:
            text: Document text to extract from
            extraction_template: Template guiding extraction
            output_format: Format for extracted data ("json" or "markdown")
            
        Returns:
            Dict containing extracted data
        """
        prompt = f"""
        {extraction_template}
        
        TEXT TO ANALYZE:
        ```
        {text}
        ```
        
        Extract the requested information in {output_format} format.
        """
        
        response = self.generate_response(
            prompt=prompt,
            system_prompt="You are a precise document analysis assistant. Extract only the requested information in the correct format.",
            max_tokens=2000
        )
        
        # Process and clean the response
        result = response["content"]
        
        # If JSON expected, parse it
        if output_format == "json" and "```json" in result:
            import json
            import re
            
            # Extract JSON block
            json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    return {"data": parsed_json, "raw_response": result}
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON", "raw_response": result}
                    
        return {"data": result, "raw_response": result}
