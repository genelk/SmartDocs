"""
Unit tests for the ClaudeInterface class.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

import anthropic

from src.models.claude_interface import ClaudeInterface

class TestClaudeInterface(unittest.TestCase):
    """Test cases for ClaudeInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Save original environment variable if it exists
        self.original_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Set a test API key
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"
        
        # Create the interface
        self.claude = ClaudeInterface()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original environment variable
        if self.original_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.original_api_key
        else:
            del os.environ["ANTHROPIC_API_KEY"]
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        claude = ClaudeInterface(api_key="explicit_api_key")
        self.assertEqual(claude.api_key, "explicit_api_key")
    
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment variable."""
        claude = ClaudeInterface()
        self.assertEqual(claude.api_key, "test_api_key")
    
    def test_init_with_no_api_key(self):
        """Test initialization with no API key."""
        # Remove API key from environment
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        
        # Expect ValueError
        with self.assertRaises(ValueError):
            ClaudeInterface()
    
    @patch("anthropic.Anthropic")
    def test_generate_response(self, mock_anthropic):
        """Test generating a response."""
        # Mock the Messages.create method
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.id = "resp_123456"
        
        mock_client.messages.create.return_value = mock_response
        
        # Test generating a response
        result = self.claude.generate_response(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            max_tokens=100
        )
        
        # Verify the result
        self.assertEqual(result["content"], "Test response")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 20)
        self.assertEqual(result["model"], "claude-3-sonnet-20240229")
        self.assertEqual(result["id"], "resp_123456")
        
        # Verify the method was called correctly
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            system="Test system prompt",
            messages=[
                {"role": "user", "content": "Test prompt"}
            ]
        )
    
    @patch("anthropic.Anthropic")
    def test_generate_response_error(self, mock_anthropic):
        """Test generating a response with an error."""
        # Mock the Messages.create method to raise an exception
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_client.messages.create.side_effect = Exception("Test error")
        
        # Test generating a response
        result = self.claude.generate_response(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            max_tokens=100
        )
        
        # Verify the result
        self.assertEqual(result["content"], "")
        self.assertEqual(result["error"], "Test error")
    
    @patch.object(ClaudeInterface, "generate_response")
    def test_extract_structured_data(self, mock_generate_response):
        """Test extracting structured data."""
        # Mock the generate_response method
        mock_generate_response.return_value = {
            "content": """
            ```json
            {
                "key1": "value1",
                "key2": "value2"
            }
            ```
            """
        }
        
        # Test extracting structured data
        result = self.claude.extract_structured_data(
            text="Test text",
            extraction_template="Test template",
            output_format="json"
        )
        
        # Verify the result
        self.assertEqual(result["data"]["key1"], "value1")
        self.assertEqual(result["data"]["key2"], "value2")
        
        # Verify the method was called correctly
        mock_generate_response.assert_called_once()
        self.assertTrue("Test template" in mock_generate_response.call_args[1]["prompt"])
        self.assertTrue("Test text" in mock_generate_response.call_args[1]["prompt"])
    
    @patch.object(ClaudeInterface, "generate_response")
    def test_extract_structured_data_invalid_json(self, mock_generate_response):
        """Test extracting structured data with invalid JSON."""
        # Mock the generate_response method with invalid JSON
        mock_generate_response.return_value = {
            "content": """
            ```json
            {
                "key1": "value1",
                "key2": 
            }
            ```
            """
        }
        
        # Test extracting structured data
        result = self.claude.extract_structured_data(
            text="Test text",
            extraction_template="Test template",
            output_format="json"
        )
        
        # Verify the result contains an error
        self.assertTrue("error" in result)
        
        # Verify raw response is included
        self.assertTrue("raw_response" in result)
    
    @patch.object(ClaudeInterface, "generate_response")
    def test_extract_structured_data_markdown(self, mock_generate_response):
        """Test extracting structured data as markdown."""
        # Mock the generate_response method
        mock_generate_response.return_value = {
            "content": "# Heading\n\n- Item 1\n- Item 2"
        }
        
        # Test extracting structured data
        result = self.claude.extract_structured_data(
            text="Test text",
            extraction_template="Test template",
            output_format="markdown"
        )
        
        # Verify the result
        self.assertTrue("# Heading" in result["data"])
        self.assertTrue("- Item 1" in result["data"])


if __name__ == "__main__":
    unittest.main()
