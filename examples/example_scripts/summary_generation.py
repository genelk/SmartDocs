"""
Example script demonstrating document summarization functionality.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.document_loader import DocumentLoader
from src.models.model_factory import ModelFactory
from src.summarizer import Summarizer

def main():
    """Run the summary generation example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate summaries for a document")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--length", choices=["very_short", "short", "medium", "long", "very_long"], 
                        default="medium", help="Summary length")
    parser.add_argument("--focus", nargs="+", help="Focus areas for the summary")
    parser.add_argument("--api-key", help="Claude API key (or set ANTHROPIC_API_KEY env variable)")
    parser.add_argument("--executive", action="store_true", help="Generate an executive summary")
    parser.add_argument("--sections", action="store_true", help="Generate section summaries")
    parser.add_argument("--output", help="Output file path (default: print to console)")
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    try:
        # Initialize components
        factory = ModelFactory()
        claude = factory.get_llm(provider="claude", api_key=args.api_key)
        doc_loader = DocumentLoader()
        summarizer = Summarizer(claude_interface=claude)
        
        print(f"Loading document: {args.file_path}")
        document = doc_loader.load_document(args.file_path)
        
        print(f"Document type: {document['document_type']}")
        print(f"Document size: {len(document['full_text'])} characters")
        
        # Generate requested summaries
        outputs = []
        
        if args.executive:
            print("Generating executive summary...")
            result = summarizer.generate_executive_summary(document)
            outputs.append(("Executive Summary", result["executive_summary"]))
            print(f"  Token usage: {result['usage']}")
        
        if args.sections:
            print("Generating section summaries...")
            sections = summarizer.generate_section_summaries(document)
            
            if sections:
                section_output = "# Section Summaries\n\n"
                for section in sections:
                    section_output += f"## {section['header']}\n\n{section['summary']}\n\n"
                outputs.append(("Section Summaries", section_output))
                print(f"  Found and summarized {len(sections)} sections")
            else:
                print("  No clear sections detected in the document")
        
        # Generate main document summary
        print(f"Generating {args.length} document summary...")
        result = summarizer.generate_document_summary(
            document=document,
            length=args.length,
            focus_areas=args.focus
        )
        
        outputs.append(("Document Summary", result["summary"]))
        print(f"  Token usage: {result['usage']}")
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for title, content in outputs:
                    f.write(f"# {title}\n\n")
                    f.write(f"{content}\n\n")
                    f.write("-" * 80 + "\n\n")
            print(f"Output written to: {args.output}")
        else:
            # Print to console
            for title, content in outputs:
                print("\n" + "=" * 80)
                print(f"# {title}")
                print("=" * 80 + "\n")
                print(content)
                print("\n" + "-" * 80 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
