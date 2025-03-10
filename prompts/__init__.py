"""
Prompt templates for SmartDocs document analysis.
"""

from .summary_templates import get_document_summary_template, get_executive_summary_template
from .extraction_templates import get_entity_extraction_template, get_data_point_extraction_template
from .qa_templates import get_basic_qa_template, get_factual_qa_template

__all__ = [
    "get_document_summary_template", 
    "get_executive_summary_template",
    "get_entity_extraction_template",
    "get_data_point_extraction_template",
    "get_basic_qa_template",
    "get_factual_qa_template"
]
