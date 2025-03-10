"""
Example script demonstrating document question-answering functionality.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.document_loader import DocumentLoader
from src.models.model_factory import ModelFactory
from prompts.qa_templates import (
    get_basic_qa_template,
    get_factual_qa_template,
    get_analytical_qa_template,
    get_followup_qa_template
)

def main():
    """Run the document Q&A example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ask questions about a document")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--question", "-q", required=True, help="Question to ask about the document")
    parser.add_argument("--mode", choices=["basic", "factual", "analytical", "followup"],
                        default="basic", help="Question-answering mode")
    parser.add_argument("--previous-question", help="Previous question (for followup mode)")
    parser.add_argument("--previous-answer", help="Previous answer (for followup mode)")
    parser.add_argument("--api-key", help="Claude API key (or set ANTHROPIC_API_KEY env variable)")
    parser.add_argument("--output", help="Output file path (default: print to console)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    # Validate followup mode requirements
    if args.mode == "followup" and (not args.previous_question or not args.previous_answer):
        print("Error: Followup mode requires --previous-question and --previous-answer")
        return 1
    
    try:
        # Initialize components
        factory = ModelFactory()
        claude = factory.get_llm(provider="claude", api_key=args.api_key)
        doc_loader = DocumentLoader()
        
        print(f"Loading document: {args.file_path}")
        document = doc_loader.load_document(args.file_path)
        
        print(f"Document type: {document['document_type']}")
        print(f"Document size: {len(document['full_text'])} characters")
        
        # Get document text, limiting to a reasonable size
        doc_text = document.get("full_text", "")
        if len(doc_text) > 25000:
            print("Note: Document is large, truncating to first 25000 characters")
            doc_text = doc_text[:25000]
        
        # Interactive mode
        if args.interactive:
            run_interactive_qa(claude, doc_text, args.mode)
            return 0
        
        # Single question mode
        print(f"Question: {args.question}")
        print(f"Mode: {args.mode}")
        
        # Get appropriate template
        if args.mode == "basic":
            template = get_basic_qa_template()
        elif args.mode == "factual":
            template = get_factual_qa_template()
        elif args.mode == "analytical":
            template = get_analytical_qa_template()
        elif args.mode == "followup":
            template = get_followup_qa_template()
        else:
            template = get_basic_qa_template()
        
        # Format prompt based on mode
        if args.mode == "followup":
            prompt = template.format(
                question=args.question,
                document_text=doc_text,
                previous_question=args.previous_question,
                previous_answer=args.previous_answer
            )
        else:
            prompt = template.format(
                question=args.question,
                document_text=doc_text
            )
        
        # Get system prompt based on mode
        system_prompt = get_system_prompt(args.mode)
        
        # Generate response
        print("Generating answer...")
        response = claude.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1000
        )
        
        answer = response["content"]
        print(f"Token usage: {response.get('usage', {})}")
        
        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"Question: {args.question}\n\n")
                f.write(f"Answer:\n{answer}")
            print(f"Output written to: {args.output}")
        else:
            # Print to console
            print("\n" + "=" * 80)
            print("Answer:")
            print("=" * 80 + "\n")
            print(answer)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

def run_interactive_qa(claude, doc_text, initial_mode="basic"):
    """Run interactive Q&A session.
    
    Args:
        claude: Claude interface
        doc_text: Document text
        initial_mode: Initial Q&A mode
    """
    print("\n" + "=" * 80)
    print("Interactive Document Q&A Mode")
    print("=" * 80)
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    print("You can change modes by typing 'mode:basic', 'mode:factual', or 'mode:analytical'")
    print("=" * 80 + "\n")
    
    mode = initial_mode
    previous_question = None
    previous_answer = None
    
    while True:
        # Get user question
        question = input("\nQuestion: ").strip()
        
        # Check for exit command
        if question.lower() in ["exit", "quit"]:
            break
        
        # Check for mode change
        if question.lower().startswith("mode:"):
            new_mode = question.lower().split(":", 1)[1].strip()
            if new_mode in ["basic", "factual", "analytical"]:
                mode = new_mode
                print(f"Mode changed to: {mode}")
            else:
                print(f"Invalid mode: {new_mode}")
            continue
        
        # Skip empty questions
        if not question:
            continue
        
        # Determine if this is a followup question
        current_mode = mode
        if previous_question and "followup" not in question.lower():
            # Ask if this is a followup question
            is_followup = input("Is this a follow-up to your previous question? (y/n): ").strip().lower()
            if is_followup in ["y", "yes"]:
                current_mode = "followup"
        
        # Get appropriate template
        if current_mode == "basic":
            template = get_basic_qa_template()
            prompt = template.format(question=question, document_text=doc_text)
        elif current_mode == "factual":
            template = get_factual_qa_template()
            prompt = template.format(question=question, document_text=doc_text)
        elif current_mode == "analytical":
            template = get_analytical_qa_template()
            prompt = template.format(question=question, document_text=doc_text)
        elif current_mode == "followup":
            template = get_followup_qa_template()
            prompt = template.format(
                question=question,
                document_text=doc_text,
                previous_question=previous_question,
                previous_answer=previous_answer
            )
        
        # Get system prompt
        system_prompt = get_system_prompt(current_mode)
        
        # Generate response
        print("Generating answer...")
        response = claude.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1000
        )
        
        answer = response["content"]
        
        # Print answer
        print("\n" + "-" * 80)
        print("Answer:")
        print("-" * 80 + "\n")
        print(answer)
        
        # Update previous Q&A for potential follow-up
        previous_question = question
        previous_answer = answer

def get_system_prompt(mode):
    """Get an appropriate system prompt based on Q&A mode.
    
    Args:
        mode: Q&A mode
        
    Returns:
        System prompt
    """
    if mode == "basic":
        return "You are a document analysis assistant. Answer questions directly based on the information in the document."
    elif mode == "factual":
        return "You are a precise document analysis assistant. Provide only factual information directly stated in the document. Always cite your sources from the document."
    elif mode == "analytical":
        return "You are an analytical document assistant. Provide thoughtful analysis of the document content, considering multiple perspectives and nuances."
    elif mode == "followup":
        return "You are a conversational document assistant. Answer follow-up questions in the context of the previous exchange, while maintaining focus on the document content."
    else:
        return "You are a document assistant. Answer questions based on the information in the document."

if __name__ == "__main__":
    sys.exit(main())
