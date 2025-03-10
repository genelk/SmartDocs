# SmartDocs

A powerful document analysis and summarization tool that leverages Claude API and Hugging Face models to extract insights from documents.

![SmartDocs Banner](https://raw.githubusercontent.com/yourusername/smartdocs/main/assets/banner.png)

## Features

- **Multi-format Support**: Process PDF, DOCX, TXT, and Markdown files
- **Smart Summarization**: Generate concise, customizable summaries at multiple levels
- **Information Extraction**: Pull out structured data from unstructured documents
- **Entity Recognition**: Identify people, organizations, and key entities using Hugging Face models
- **Interactive Q&A**: Ask questions about document content and get accurate answers
- **Semantic Search**: Find information across multiple documents

## Architecture

SmartDocs combines the power of Claude's advanced language understanding with specialized Hugging Face models for specific NLP tasks:

- **Claude API**: Handles summarization, information extraction, and Q&A
- **Hugging Face Models**: Provide entity recognition, document classification, and text embeddings
- **Document Processing**: Smart chunking and preprocessing for optimal analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/genelk/smartdocs.git
cd smartdocs

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## API Key Setup

### Claude API Key (Required for core functionality)

This project uses Anthropic's Claude API for document analysis, summarization, and question answering. You will need your own Claude API key to run this project.

**How to get a Claude API key:**

1. Visit [Anthropic's API Console](https://console.anthropic.com)
2. Create an account or log in
3. Navigate to the "API Keys" section
4. Create a new API key
5. Add payment information (plans start at approximately $5/month)

**Setting up your API key:**

```bash
# Set as environment variable
export ANTHROPIC_API_KEY="your_claude_api_key"  # On Linux/macOS
set ANTHROPIC_API_KEY=your_claude_api_key       # On Windows CMD
$env:ANTHROPIC_API_KEY="your_claude_api_key"    # On Windows PowerShell

# OR pass directly to scripts with --api-key parameter
python examples/example_scripts/summary_generation.py sample.pdf --api-key your_claude_api_key
```

⚠️ **Important:** The Claude API is a paid service. Usage costs will depend on the volume of text processed and the specific model used. Be mindful of your usage to avoid unexpected charges.

## Usage

### Web Interface

```bash
streamlit run src/app.py
```

Then open your browser to http://localhost:8501

### Python API

```python
from smartdocs.document_loader import DocumentLoader
from smartdocs.models.claude_interface import ClaudeInterface
from smartdocs.summarizer import generate_summary

# Load a document
loader = DocumentLoader()
document = loader.load_document("path/to/document.pdf")

# Generate a summary
claude = ClaudeInterface(api_key="your_api_key")
summary = generate_summary(document, claude, length="medium")

print(summary)
```

## Example Use Cases

- **Research Paper Analysis**: Quickly understand key findings and methodology
- **Contract Review**: Extract important dates, obligations, and entities
- **Report Summarization**: Create executive summaries of lengthy reports
- **Data Extraction**: Pull structured data from unstructured documents
- **Knowledge Base Creation**: Extract and organize information from document collections

## Roadmap

- [ ] Data visualization capabilities for extracted numerical information
- [ ] Multi-document comparison and analysis
- [ ] Fine-tuned specialized models for different document types
- [ ] Export capabilities to various formats
- [ ] Browser extension for analyzing web content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Anthropic for the Claude API
- Hugging Face for their open-source models and transformers library
- PyMuPDF, python-docx and other document processing libraries
