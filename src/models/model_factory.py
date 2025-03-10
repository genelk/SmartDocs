"""
Factory pattern implementation for model selection.
"""
import os
from typing import Dict, Optional, Any, Union

from .claude_interface import ClaudeInterface
from .huggingface_models import HuggingFaceManager

class ModelFactory:
    """Factory for creating and managing model instances."""
    
    def __init__(self):
        """Initialize the model factory."""
        self._models = {}
        self._default_models = {
            "llm": "claude",
            "embeddings": "huggingface",
            "ner": "huggingface",
            "classification": "huggingface"
        }
    
    def get_llm(self, provider: str = None, **kwargs) -> Union[ClaudeInterface, Any]:
        """Get a large language model instance.
        
        Args:
            provider: Provider name ('claude' by default)
            **kwargs: Additional config params for the model
            
        Returns:
            Model instance
        """
        provider = provider or self._default_models["llm"]
        
        # Create singleton instance if it doesn't exist
        model_key = f"llm_{provider}"
        if model_key not in self._models:
            if provider == "claude":
                api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
                model = kwargs.get("model", "claude-3-sonnet-20240229")
                self._models[model_key] = ClaudeInterface(api_key=api_key, model=model)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return self._models[model_key]
    
    def get_huggingface_manager(self, **kwargs) -> HuggingFaceManager:
        """Get a Hugging Face models manager instance.
        
        Args:
            **kwargs: Additional config params for the manager
            
        Returns:
            HuggingFaceManager instance
        """
        # Create singleton instance if it doesn't exist
        model_key = "huggingface_manager"
        if model_key not in self._models:
            cache_dir = kwargs.get("cache_dir")
            device = kwargs.get("device", "cpu")
            self._models[model_key] = HuggingFaceManager(cache_dir=cache_dir, device=device)
        
        return self._models[model_key]
    
    def get_embeddings_model(self, provider: str = None, **kwargs) -> Any:
        """Get a text embeddings model.
        
        Args:
            provider: Provider name ('huggingface' by default)
            **kwargs: Additional config params for the model
            
        Returns:
            Embeddings model instance or interface
        """
        provider = provider or self._default_models["embeddings"]
        
        if provider == "huggingface":
            hf_manager = self.get_huggingface_manager(**kwargs)
            model_name = kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            return hf_manager.get_embeddings_model(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")
    
    def get_ner_model(self, provider: str = None, **kwargs) -> Any:
        """Get a named entity recognition model.
        
        Args:
            provider: Provider name ('huggingface' by default)
            **kwargs: Additional config params for the model
            
        Returns:
            NER model instance or interface
        """
        provider = provider or self._default_models["ner"]
        
        if provider == "huggingface":
            hf_manager = self.get_huggingface_manager(**kwargs)
            model_name = kwargs.get("model_name", "dslim/bert-base-NER")
            return hf_manager.get_ner_model(model_name=model_name)
        else:
            raise ValueError(f"Unsupported NER provider: {provider}")
    
    def set_default_provider(self, task: str, provider: str) -> None:
        """Set the default provider for a specific task.
        
        Args:
            task: Task name ('llm', 'embeddings', 'ner', or 'classification')
            provider: Provider name to use by default
        """
        if task not in self._default_models:
            valid_tasks = list(self._default_models.keys())
            raise ValueError(f"Invalid task: {task}. Valid tasks are: {valid_tasks}")
        
        self._default_models[task] = provider
