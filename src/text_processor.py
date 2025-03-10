"""
Text processing utilities for document preprocessing and chunking.
"""
import re
from typing import List, Dict, Any, Optional

class TextProcessor:
    """Process and manipulate document text for optimal analysis."""
    
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """Initialize text processor.
        
        Args:
            chunk_size: Maximum character length for document chunks
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize newlines
        text = text.replace('\r\n', '\n')
        
        return text.strip()
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple regex for sentence splitting - can be improved
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for processing.
        
        Args:
            text: Input text
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk objects with text and metadata
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end of the text, try to break at a paragraph or sentence
            if end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find sentence break
                    sentence_break = max(
                        text.rfind(". ", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("? ", start, end)
                    )
                    if sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
            
            chunk_text = text[start:end].strip()
            
            chunk_data = {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "word_count": len(chunk_text.split())
            }
            
            # Add any provided metadata
            if metadata:
                chunk_data["metadata"] = metadata
                
            chunks.append(chunk_data)
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap
            
        return chunks
    
    def find_section_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential section headers from text.
        
        Args:
            text: Input text
            
        Returns:
            List of section headers with positions
        """
        # Look for common section header patterns
        patterns = [
            # Numbered headers (1. Introduction, 1.1 Background, etc.)
            r'^(?:\d+\.)+\s+([A-Z][^\n]+)$',
            # All caps headers
            r'^([A-Z][A-Z\s]+[A-Z])$',
            # Title case headers with colon
            r'^([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+):',
            # Common section names
            r'^(Introduction|Abstract|Methodology|Results|Discussion|Conclusion|References)$'
        ]
        
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check each pattern
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    headers.append({
                        "text": line,
                        "matched_text": match.group(1),
                        "line_number": i,
                        "char_position": text.find(line)
                    })
                    break
        
        return headers
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Attempt to identify and extract tables from text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential tables with row/column information
        """
        # Look for common table patterns in plain text
        tables = []
        
        # Find potential table blocks (text with consistent spacing/structure)
        lines = text.split('\n')
        table_start = None
        
        # Pattern for table row with multiple columns separated by whitespace
        table_row_pattern = r'^\s*\S+(?:\s{2,}\S+){2,}\s*$'
        
        for i, line in enumerate(lines):
            is_table_row = re.match(table_row_pattern, line) is not None
            
            if is_table_row and table_start is None:
                table_start = i
            elif not is_table_row and table_start is not None:
                # We've reached the end of a potential table
                table_lines = lines[table_start:i]
                if len(table_lines) >= 3:  # At least header + separator + one row
                    # Process the table
                    table = {
                        "start_line": table_start,
                        "end_line": i-1,
                        "rows": [],
                        "raw_text": '\n'.join(table_lines)
                    }
                    
                    # Simple column detection by splitting on multiple spaces
                    for row in table_lines:
                        table["rows"].append(re.split(r'\s{2,}', row.strip()))
                    
                    # Get header and data rows
                    if len(table["rows"]) > 0:
                        table["headers"] = table["rows"][0]
                        table["data"] = table["rows"][1:]
                        
                    tables.append(table)
                
                table_start = None
        
        # Check if the last lines form a table
        if table_start is not None and (len(lines) - table_start) >= 3:
            table_lines = lines[table_start:]
            table = {
                "start_line": table_start,
                "end_line": len(lines)-1,
                "rows": [],
                "raw_text": '\n'.join(table_lines)
            }
            
            for row in table_lines:
                table["rows"].append(re.split(r'\s{2,}', row.strip()))
            
            if len(table["rows"]) > 0:
                table["headers"] = table["rows"][0]
                table["data"] = table["rows"][1:]
                
            tables.append(table)
        
        return tables
