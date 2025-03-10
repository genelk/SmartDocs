# src/document_loader.py
import os
import fitz  # PyMuPDF
import docx
import markdown
from typing import Dict, List, Optional, Any, Tuple

class DocumentLoader:
    """Load and preprocess documents from various formats."""
    
    def __init__(self, chunk_size: int = 8000, chunk_overlap: int = 200):
        """Initialize document loader.
        
        Args:
            chunk_size: Maximum character length for document chunks
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load document from file path.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dict containing document text, metadata, and extracted sections
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.docx':
            return self._load_docx(file_path)
        elif file_ext == '.md':
            return self._load_markdown(file_path)
        elif file_ext == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Load and parse PDF document."""
        doc = fitz.open(file_path)
        text = ""
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "page_count": len(doc)
        }
        
        # Extract text from each page
        pages = []
        for i, page in enumerate(doc):
            page_text = page.get_text()
            pages.append({
                "page_num": i+1,
                "text": page_text,
                "word_count": len(page_text.split())
            })
            text += page_text + "\n\n"
            
        # Create chunks
        chunks = self._create_chunks(text)
        
        return {
            "document_type": "pdf",
            "file_path": file_path,
            "metadata": metadata,
            "full_text": text,
            "pages": pages,
            "chunks": chunks
        }
    
    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """Load and parse DOCX document."""
        doc = docx.Document(file_path)
        text = ""
        
        # Extract paragraphs
        paragraphs = []
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                paragraphs.append({
                    "id": i,
                    "text": para.text,
                    "style": para.style.name
                })
                text += para.text + "\n\n"
        
        # Create chunks
        chunks = self._create_chunks(text)
        
        return {
            "document_type": "docx",
            "file_path": file_path,
            "metadata": {
                "title": doc.core_properties.title or "",
                "author": doc.core_properties.author or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
                "paragraph_count": len(paragraphs)
            },
            "full_text": text,
            "paragraphs": paragraphs,
            "chunks": chunks
        }
    
    # Methods for other file formats would be implemented similarly
    
    def _create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
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
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "word_count": len(chunk_text.split())
            })
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap
            
        return chunks
