"""
Prompt templates for information extraction from documents.
"""

def get_entity_extraction_template() -> str:
    """Get a prompt template for named entity extraction.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Extract all named entities from the following document:
    
    ```
    {document_text}
    ```
    
    Please identify and categorize the following types of entities:
    1. People (including full names and titles/roles where available)
    2. Organizations (companies, institutions, agencies, etc.)
    3. Locations (cities, countries, addresses, etc.)
    4. Dates and times
    
    For each entity, include a brief description or context if available in the document.
    
    Format the output as valid JSON with the following structure:
    {
        "people": [
            {"name": "Person name", "title": "Title/role (if any)", "context": "Brief context"}
        ],
        "organizations": [
            {"name": "Organization name", "type": "Type of organization (if known)", "context": "Brief context"}
        ],
        "locations": [
            {"name": "Location name", "type": "Type of location", "context": "Brief context"}
        ],
        "dates": [
            {"date": "Date or time mentioned", "context": "Brief context"}
        ]
    }
    """
    
    return template


def get_data_point_extraction_template() -> str:
    """Get a prompt template for extracting numerical data and statistics.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Extract all numerical data points and statistics from the following document:
    
    ```
    {document_text}
    ```
    
    Focus on extracting:
    1. Monetary values (costs, prices, budgets, revenues, etc.)
    2. Percentages and ratios
    3. Quantities and measurements
    4. Dates and time periods
    5. Statistical figures and metrics
    
    For each data point, include:
    - The exact value as mentioned in the text
    - The category or type of data
    - The context (what the number refers to)
    - The unit (if applicable)
    
    Format the output as valid JSON with the following structure:
    {
        "monetary_values": [
            {"value": "Value as text", "amount": number, "currency": "Currency code", "context": "Brief context"}
        ],
        "percentages": [
            {"value": "Value as text", "percentage": number, "context": "Brief context"}
        ],
        "quantities": [
            {"value": "Value as text", "quantity": number, "unit": "Unit of measurement", "context": "Brief context"}
        ],
        "dates": [
            {"value": "Date as text", "context": "Brief context"}
        ],
        "statistics": [
            {"value": "Value as text", "metric": "Name of metric", "context": "Brief context"}
        ]
    }
    
    If you're unsure about a value, include it and note your uncertainty in the context.
    """
    
    return template


def get_key_points_extraction_template(num_points: int = 5) -> str:
    """Get a prompt template for extracting key points or takeaways.
    
    Args:
        num_points: Number of key points to extract
        
    Returns:
        Formatted prompt template
    """
    template = f"""
    Extract the {num_points} most important points or takeaways from the following document:
    
    ```
    {{document_text}}
    ```
    
    For each key point:
    1. Provide a concise 1-sentence summary of the point
    2. Include a brief explanation or supporting detail
    3. Note where in the document this point is discussed (if possible)
    
    Format the output as valid JSON with the following structure:
    {{
        "key_points": [
            {{
                "point": "Concise statement of key point",
                "explanation": "Brief explanation with additional context or detail",
                "location": "Section or area of document where discussed (optional)"
            }}
        ]
    }}
    
    Focus on the most significant, substantive information rather than minor details.
    """
    
    return template


def get_contract_extraction_template() -> str:
    """Get a prompt template for extracting contract information.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Extract the key contract terms and clauses from the following document:
    
    ```
    {document_text}
    ```
    
    Focus on identifying:
    1. Parties involved (names, roles, contact information)
    2. Effective dates and duration (start date, end date, term length)
    3. Payment terms and amounts (payment schedule, fees, currencies)
    4. Deliverables and obligations (what each party must provide)
    5. Termination conditions (how the contract can be ended)
    6. Liabilities and warranties (responsibilities and guarantees)
    7. Special clauses or conditions (non-compete, confidentiality, etc.)
    
    Format the output as valid JSON with appropriate fields for each type of information.
    Be precise and include exact dates, amounts, and specific conditions as stated in the document.
    """
    
    return template


def get_research_findings_extraction_template() -> str:
    """Get a prompt template for extracting research findings.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Extract the key research findings from the following document:
    
    ```
    {document_text}
    ```
    
    Focus on identifying:
    1. Study objectives/research questions
    2. Methodology used (participants, procedures, measures)
    3. Main results and findings (data, statistics, outcomes)
    4. Statistical significance (p-values, confidence intervals, effect sizes)
    5. Limitations mentioned
    6. Conclusions and implications
    
    Format the output as valid JSON with the following structure:
    {
        "objectives": [
            {"objective": "Description of research objective or question"}
        ],
        "methodology": {
            "design": "Study design description",
            "participants": "Information about study participants",
            "procedures": "Description of procedures used",
            "measures": "Measures or instruments used"
        },
        "results": [
            {"finding": "Description of finding", "statistics": "Statistical information if available"}
        ],
        "limitations": [
            {"limitation": "Description of study limitation"}
        ],
        "conclusions": [
            {"conclusion": "Description of conclusion or implication"}
        ]
    }
    
    Be precise and include specific details, statistics, and quoted findings when available.
    """
    
    return template


def get_product_specs_extraction_template() -> str:
    """Get a prompt template for extracting product specifications.
    
    Returns:
        Formatted prompt template
    """
    template = """
    Extract product specifications from the following document:
    
    ```
    {document_text}
    ```
    
    Focus on identifying:
    1. Product name and model
    2. Technical specifications and parameters
    3. Features and capabilities
    4. Dimensions and physical characteristics
    5. Compatibility and requirements
    6. Pricing information (if available)
    
    Format the output as valid JSON with the following structure:
    {
        "product_info": {
            "name": "Product name",
            "model": "Model number",
            "manufacturer": "Manufacturer name",
            "release_date": "Release date if mentioned"
        },
        "technical_specs": [
            {"spec_name": "Name of specification", "value": "Value", "unit": "Unit if applicable"}
        ],
        "features": [
            {"feature": "Description of feature"}
        ],
        "dimensions": {
            "height": "Height with unit",
            "width": "Width with unit",
            "depth": "Depth with unit",
            "weight": "Weight with unit"
        },
        "compatibility": [
            {"requirement": "Compatibility requirement"}
        ],
        "pricing": [
            {"price_type": "Type of price (MSRP, sale, etc.)", "amount": "Amount", "currency": "Currency"}
        ]
    }
    
    Be precise and include exact specifications, measurements, and features as stated in the document.
    """
    
    return template


def get_custom_extraction_template(extraction_instructions: str) -> str:
    """Get a customized prompt template based on specific extraction instructions.
    
    Args:
        extraction_instructions: Specific instructions for what to extract
        
    Returns:
        Formatted prompt template
    """
    template = f"""
    Extract the following information from this document:
    
    {extraction_instructions}
    
    DOCUMENT:
    ```
    {{document_text}}
    ```
    
    Format the output as structured JSON with appropriate fields for each type of requested information.
    Be precise and include exact details as stated in the document.
    """
    
    return template
