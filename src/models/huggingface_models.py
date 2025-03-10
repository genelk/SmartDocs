# src/models/huggingface_models.py
from typing import Dict, List, Any, Optional, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModel,
    pipeline
)

class HuggingFaceManager:
    """Manager for Hugging Face models for specialized NLP tasks."""
    
    def __init__(self, cache_dir: Optional[str] = None, device: str = "cpu"):
        """Initialize HuggingFace models manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
            device: Device to run models on ("cpu" or "cuda")
        """
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.models = {}
        self.tokenizers = {}
        
    def get_document_classifier(self, model_name: str = "facebook/bart-large-mnli"):
        """Load and cache document classification model.
        
        Args:
            model_name: Hugging Face model name
            
        Returns:
            Classification pipeline
        """
        key = f"classifier_{model_name}"
        if key not in self.models:
            self.models[key] = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.device
            )
        return self.models[key]
    
    def classify_document(self, 
                         text: str, 
                         categories: List[str], 
                         multi_label: bool = False) -> Dict[str, Any]:
        """Classify document text into provided categories.
        
        Args:
            text: Document text
            categories: List of categories to classify into
            multi_label: Whether multiple labels can apply
            
        Returns:
            Dict with classification results
        """
        classifier = self.get_document_classifier()
        
        # Truncate text if too long
        text = text[:1024*3]  # Avoid token length issues
        
        result = classifier(text, categories, multi_label=multi_label)
        
        return {
            "labels": result["labels"],
            "scores": result["scores"],
            "top_category": result["labels"][0],
            "top_score": result["scores"][0]
        }
    
    def get_ner_model(self, model_name: str = "dslim/bert-base-NER"):
        """Load and cache named entity recognition model."""
        key = f"ner_{model_name}"
        if key not in self.models:
            self.models[key] = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=self.device
            )
        return self.models[key]
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from text.
        
        Args:
            text: Document text
            
        Returns:
            Dict with entities by type
        """
        ner_model = self.get_ner_model()
        
        # Process in chunks to avoid token length issues
        chunks = [text[i:i+512] for i in range(0, len(text), 384)]  # overlap of 128
        
        all_entities = []
        for chunk in chunks:
            entities = ner_model(chunk)
            all_entities.extend(entities)
        
        # Group by entity type
        entity_groups = {}
        for entity in all_entities:
            entity_type = entity["entity_group"]
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            
            # Add if not duplicate
            word = entity["word"]
            if not any(e["word"] == word for e in entity_groups[entity_type]):
                entity_groups[entity_type].append(entity)
        
        return entity_groups
    
    def get_embeddings_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load and cache text embeddings model."""
        key = f"embeddings_{model_name}"
        if key not in self.models:
            self.tokenizers[key] = AutoTokenizer.from_pretrained(model_name)
            self.models[key] = AutoModel.from_pretrained(model_name)
            self.models[key] = self.models[key].to(self.device)
        return self.models[key], self.tokenizers[key]
    
    def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for text chunks.
        
        Args:
            texts: List of text chunks
            
        Returns:
            Dict with embeddings and metadata
        """
        model, tokenizer = self.get_embeddings_model()
        
        embeddings = []
        for text in texts:
            # Truncate if needed
            text = text[:512]
            
            # Tokenize and convert to tensor
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                output = model(**inputs)
            
            # Use mean pooling
            embedding = output.last_hidden_state.mean(dim=1)
            embeddings.append(embedding.cpu().numpy()[0])
        
        return {
            "embeddings": embeddings,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": embeddings[0].shape[0] if embeddings else 0
        }
