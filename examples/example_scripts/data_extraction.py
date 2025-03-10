"""
Example script demonstrating data extraction functionality.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.document_loader import DocumentLoader
from src.models.model_factory import ModelFactory
from src.extraction import Extractor

def main():
    """Run the data extraction example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract information from a document")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--extraction-type", "-t", choices=["entities", "data_points", "key_points", "contract", "research", "product", "custom"], 
                        default="entities", help="Type of information to extract")
    parser.add_argument("--custom-instructions", help="Custom extraction instructions (for custom type)")
    parser.add_argument("--num-points", type=int, default=5, help="Number of key points to extract")
    parser.add_argument("--api-key", help="Claude API key (or set ANTHROPIC_API_KEY env variable)")
    parser.add_argument("--output", help="Output file path (default: print to console)")
    parser.add_argument("--format", choices=["text", "json"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    # Validate custom instructions
    if args.extraction_type == "custom" and not args.custom_instructions:
        print("Error: Custom extraction type requires --custom-instructions")
        return 1
    
    try:
        # Initialize components
        factory = ModelFactory()
        claude = factory.get_llm(provider="claude", api_key=args.api_key)
        doc_loader = DocumentLoader()
        extractor = Extractor(claude_interface=claude)
        
        print(f"Loading document: {args.file_path}")
        document = doc_loader.load_document(args.file_path)
        
        print(f"Document type: {document['document_type']}")
        print(f"Document size: {len(document['full_text'])} characters")
        
        # Perform extraction based on type
        print(f"Extracting {args.extraction_type} from document...")
        
        if args.extraction_type == "entities":
            result = extractor.extract_entities(document)
            extracted_data = result["entities"]
            
        elif args.extraction_type == "data_points":
            result = extractor.extract_data_points(document)
            extracted_data = result["data_points"]
            
        elif args.extraction_type == "key_points":
            result = extractor.extract_key_points(document, num_points=args.num_points)
            extracted_data = {"key_points": result["key_points"]}
            
        else:
            # Map extraction type to information_type
            info_type_map = {
                "contract": "contract_terms",
                "research": "research_findings",
                "product": "product_specs",
                "custom": "custom"
            }
            
            information_type = info_type_map.get(args.extraction_type)
            
            result = extractor.extract_specific_information(
                document=document,
                information_type=information_type,
                custom_instructions=args.custom_instructions
            )
            
            extracted_data = result["extracted_information"]
        
        print(f"Extraction complete")
        
        # Format and output results
        if args.format == "json":
            formatted_output = json.dumps(extracted_data, indent=2)
        else:
            # Simple text format
            formatted_output = format_as_text(extracted_data)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            print(f"Output written to: {args.output}")
        else:
            # Print to console
            print("\n" + "=" * 80)
            print(f"Extracted {args.extraction_type.replace('_', ' ').title()}:")
            print("=" * 80 + "\n")
            print(formatted_output)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

def format_as_text(data, indent=0):
    """Format data as readable text.
    
    Args:
        data: Data to format (dict, list, or scalar)
        indent: Current indentation level
        
    Returns:
        Formatted text
    """
    indent_str = "  " * indent
    
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key}:")
                lines.append(format_as_text(value, indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return "\n".join(lines)
    
    elif isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(format_as_text(item, indent))
                lines.append("")  # Empty line between items
            else:
                lines.append(f"{indent_str}- {item}")
        return "\n".join(lines)
    
    else:
        return f"{indent_str}{data}"

if __name__ == "__main__":
    sys.exit(main())
