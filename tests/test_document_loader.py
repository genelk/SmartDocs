"""
Unit tests for the DocumentLoader class.
"""
import os
import unittest
from unittest.mock import patch, mock_open, MagicMock

import fitz
import docx

from src.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):
    """Test cases for DocumentLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader(chunk_size=1000, chunk_overlap=100)
        
        # Create test directory if it doesn't exist
        os.makedirs("tests/test_files", exist_ok=True)
    
    @patch("fitz.open")
    def test_load_pdf(self, mock_fitz_open):
        """Test loading a PDF document."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test PDF",
            "author": "Test Author",
            "creationDate": "D:20220101120000",
        }
        
        # Mock PDF page
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a test PDF page content."
        
        # Set up the mock document with one page
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc
        
        # Test loading the PDF
        result = self.loader._load_pdf("tests/test_files/test.pdf")
        
        # Verify the result
        self.assertEqual(result["document_type"], "pdf")
        self.assertEqual(result["metadata"]["title"], "Test PDF")
        self.assertEqual(result["metadata"]["author"], "Test Author")
        self.assertEqual(result["metadata"]["page_count"], 1)
        self.assertEqual(len(result["pages"]), 1)
        self.assertEqual(result["pages"][0]["text"], "This is a test PDF page content.")
        self.assertTrue("full_text" in result)
        self.assertTrue("chunks" in result)
    
    @patch("docx.Document")
    def test_load_docx(self, mock_docx):
        """Test loading a DOCX document."""
        # Mock DOCX document
        mock_doc = MagicMock()
        
        # Mock core properties
        mock_doc.core_properties = MagicMock()
        mock_doc.core_properties.title = "Test DOCX"
        mock_doc.core_properties.author = "Test Author"
        mock_doc.core_properties.created = "2022-01-01"
        
        # Mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "This is paragraph 1."
        mock_para1.style.name = "Normal"
        
        mock_para2 = MagicMock()
        mock_para2.text = "This is paragraph 2."
        mock_para2.style.name = "Heading 1"
        
        mock_doc.paragraphs = [mock_para1, mock_para2]
        
        # Set up the mock to return our mock document
        mock_docx.return_value = mock_doc
        
        # Test loading the DOCX
        result = self.loader._load_docx("tests/test_files/test.docx")
        
        # Verify the result
        self.assertEqual(result["document_type"], "docx")
        self.assertEqual(result["metadata"]["title"], "Test DOCX")
        self.assertEqual(result["metadata"]["author"], "Test Author")
        self.assertEqual(len(result["paragraphs"]), 2)
        self.assertEqual(result["paragraphs"][0]["text"], "This is paragraph 1.")
        self.assertEqual(result["paragraphs"][1]["style"], "Heading 1")
        self.assertTrue("full_text" in result)
        self.assertTrue("chunks" in result)
    
    @patch("builtins.open", new_callable=mock_open, read_data="This is a test text file.\nIt has multiple lines.")
    def test_load_text(self, mock_file):
        """Test loading a text file."""
        # Test loading the text file
        result = self.loader._load_text("tests/test_files/test.txt")
        
        # Verify the result
        self.assertEqual(result["document_type"], "txt")
        self.assertEqual(result["file_path"], "tests/test_files/test.txt")
        self.assertEqual(result["full_text"], "This is a test text file.\nIt has multiple lines.")
        self.assertTrue("chunks" in result)
    
    @patch("builtins.open", new_callable=mock_open, read_data="# Test Markdown\n\nThis is a **markdown** file.")
    def test_load_markdown(self, mock_file):
        """Test loading a markdown file."""
        # Test loading the markdown file
        result = self.loader._load_markdown("tests/test_files/test.md")
        
        # Verify the result
        self.assertEqual(result["document_type"], "md")
        self.assertEqual(result["file_path"], "tests/test_files/test.md")
        self.assertTrue("# Test Markdown" in result["full_text"])
        self.assertTrue("**markdown**" in result["full_text"])
        self.assertTrue("chunks" in result)
    
    def test_create_chunks(self):
        """Test creating chunks from text."""
        # Test text with paragraphs
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3 is longer and contains multiple sentences. This is the second sentence. This is the third sentence."
        
        result = self.loader._create_chunks(text)
        
        # Verify the result
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0]["chunk_id"], 0)
        self.assertEqual(result[0]["text"], "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3 is longer and contains multiple sentences. This is the second sentence. This is the third sentence.")
        
        # Test longer text that needs to be split
        long_text = "A" * 2000
        
        result = self.loader._create_chunks(long_text)
        
        # Verify the result
        self.assertTrue(len(result) > 1)
        self.assertEqual(result[0]["chunk_id"], 0)
        self.assertEqual(result[1]["chunk_id"], 1)
        self.assertEqual(len(result[0]["text"]), 1000)  # Our chunk_size is 1000
    
    def test_load_document_unsupported_format(self):
        """Test loading a document with an unsupported format."""
        with self.assertRaises(ValueError):
            self.loader.load_document("tests/test_files/test.xyz")
    
    def test_load_document_file_not_found(self):
        """Test loading a non-existent document."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_document("tests/test_files/nonexistent.pdf")


if __name__ == "__main__":
    unittest.main()
