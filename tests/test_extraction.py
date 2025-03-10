"""
Unit tests for the Extractor class.
"""
import json
import unittest
from unittest.mock import patch, MagicMock

from src.extraction import Extractor
from src.models.claude_interface import ClaudeInterface
from src.text_processor import TextProcessor

class TestExtractor(unittest.TestCase):
    """Test cases for Extractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ClaudeInterface
        self.mock_claude = MagicMock(spec=ClaudeInterface)
        
        # Create Extractor with mock Claude
        self.extractor = Extractor(claude_interface=self.mock_claude)
        
        # Sample document
        self.sample_document = {
            "file_path": "test_doc.pdf",
            "full_text": "This is a test document. It mentions John Smith who works at Acme Corp. The company is located in New York. The project costs $500,000 and will be completed by January 2023."
        }
    
    def test_extract_entities(self):
        """Test extracting entities from a document."""
        # Mock response from Claude
        mock_entities = {
            "people": [
                {"name": "John Smith", "title": "", "context": "works at Acme Corp"}
            ],
            "organizations": [
                {"name": "Acme Corp", "type": "Company", "context": "John Smith works there"}
            ],
            "locations": [
                {"name": "New York", "type": "City", "context": "location of Acme Corp"}
            ],
            "dates": [
                {"date": "January 2023", "context": "project completion date"}
            ]
        }
        
        self.mock_claude.generate_response.return_value = {
            "content": json.dumps(mock_entities),
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        # Test extracting entities
        result = self.extractor.extract_entities(self.sample_document)
        
        # Verify the result
        self.assertEqual(result["document_id"], "test_doc.pdf")
        self.assertEqual(result["entities"], mock_entities)
        self.assertTrue("usage" in result)
        
        # Verify prompt construction
        self.mock_claude.generate_response.assert_called_once()
        prompt = self.mock_claude.generate_response.call_args[1]["prompt"]
        self.assertTrue("Extract all named entities" in prompt)
        self.assertTrue(self.sample_document["full_text"] in prompt)
    
    def test_extract_data_points(self):
        """Test extracting data points from a document."""
        # Mock response from Claude
        mock_data_points = {
            "monetary_values": [
                {"value": "$500,000", "amount": 500000, "currency": "USD", "context": "project cost"}
            ],
            "percentages": [],
            "quantities": [],
            "dates": [
                {"value": "January 2023", "context": "project completion date"}
            ],
            "statistics": []
        }
        
        self.mock_claude.generate_response.return_value = {
            "content": json.dumps(mock_data_points),
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        # Test extracting data points
        result = self.extractor.extract_data_points(self.sample_document)
        
        # Verify the result
        self.assertEqual(result["document_id"], "test_doc.pdf")
        self.assertEqual(result["data_points"], mock_data_points)
        self.assertTrue("usage" in result)
        
        # Verify prompt construction
        self.mock_claude.generate_response.assert_called_once()
        prompt = self.mock_claude.generate_response.call_args[1]["prompt"]
        self.assertTrue("Extract all numerical data points" in prompt)
        self.assertTrue(self.sample_document["full_text"] in prompt)
    
    def test_extract_key_points(self):
        """Test extracting key points from a document."""
        # Mock response from Claude
        mock_key_points = {
            "key_points": [
                {
                    "point": "John Smith works at Acme Corp",
                    "explanation": "The document mentions an employee relationship",
                    "location": "Beginning of the document"
                },
                {
                    "point": "The project costs $500,000",
                    "explanation": "The document mentions the budget for a project",
                    "location": "Middle of the document"
                }
            ]
        }
        
        self.mock_claude.generate_response.return_value = {
            "content": json.dumps(mock_key_points),
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        # Test extracting key points
        result = self.extractor.extract_key_points(self.sample_document, num_points=2)
        
        # Verify the result
        self.assertEqual(result["document_id"], "test_doc.pdf")
        self.assertEqual(result["key_points"], mock_key_points["key_points"])
        self.assertTrue("usage" in result)
        
        # Verify prompt construction
        self.mock_claude.generate_response.assert_called_once()
        prompt = self.mock_claude.generate_response.call_args[1]["prompt"]
        self.assertTrue("Extract the 2 most important points" in prompt)
        self.assertTrue(self.sample_document["full_text"] in prompt)
    
    def test_extract_specific_information(self):
        """Test extracting specific information from a document."""
        # Mock response from Claude
        mock_contract_info = {
            "parties": [
                {"name": "Acme Corp", "role": "Service Provider"},
                {"name": "John Smith", "role": "Client"}
            ],
            "effective_date": "January 2023",
            "amount": "$500,000"
        }
        
        self.mock_claude.generate_response.return_value = {
            "content": json.dumps(mock_contract_info),
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        # Test extracting specific information
        result = self.extractor.extract_specific_information(
            document=self.sample_document,
            information_type="contract_terms"
        )
        
        # Verify the result
        self.assertEqual(result["document_id"], "test_doc.pdf")
        self.assertEqual(result["information_type"], "contract_terms")
        self.assertEqual(result["extracted_information"], mock_contract_info)
        self.assertTrue("usage" in result)
        
        # Verify prompt construction
        self.mock_claude.generate_response.assert_called_once()
        prompt = self.mock_claude.generate_response.call_args[1]["prompt"]
        self.assertTrue("Extract the following information" in prompt)
        self.assertTrue("contract terms" in prompt.lower())
        self.assertTrue(self.sample_document["full_text"] in prompt)
    
    def test_extract_specific_information_custom(self):
        """Test extracting custom information from a document."""
        # Mock response from Claude
        mock_custom_info = {
            "custom_field": "custom_value"
        }
        
        self.mock_claude.generate_response.return_value = {
            "content": json.dumps(mock_custom_info),
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        # Test extracting custom information
        result = self.extractor.extract_specific_information(
            document=self.sample_document,
            information_type="custom",
            custom_instructions="Extract any custom information"
        )
        
        # Verify the result
        self.assertEqual(result["document_id"], "test_doc.pdf")
        self.assertEqual(result["information_type"], "custom")
        self.assertEqual(result["extracted_information"], mock_custom_info)
        self.assertTrue("usage" in result)
        
        # Verify prompt construction
        self.mock_claude.generate_response.assert_called_once()
        prompt = self.mock_claude.generate_response.call_args[1]["prompt"]
        self.assertTrue("Extract any custom information" in prompt)
        self.assertTrue(self.sample_document["full_text"] in prompt)
    
    def test_extract_json_from_text_code_block(self):
        """Test extracting JSON from text with code block."""
        text = """
        Here's the extracted information:
        
        ```json
        {
            "key1": "value1",
            "key2": "value2"
        }
        ```
        """
        
        result = self.extractor._extract_json_from_text(text)
        
        # Verify the result
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], "value2")
    
    def test_extract_json_from_text_direct(self):
        """Test extracting JSON from text without code block."""
        text = """
        {
            "key1": "value1",
            "key2": "value2"
        }
        """
        
        result = self.extractor._extract_json_from_text(text)
        
        # Verify the result
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], "value2")
    
    def test_extract_json_from_text_partial(self):
        """Test extracting JSON from text with surrounding text."""
        text = """
        Here's the extracted information:
        
        {
            "key1": "value1",
            "key2": "value2"
        }
        
        Let me know if you need anything else.
        """
        
        result = self.extractor._extract_json_from_text(text)
        
        # Verify the result
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], "value2")
    
    def test_extract_json_from_text_invalid(self):
        """Test extracting JSON from text with invalid JSON."""
        text = """
        Here's the extracted information:
        
        This is not valid JSON.
        """
        
        result = self.extractor._extract_json_from_text(text)
        
        # Verify the result is an empty dict
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
