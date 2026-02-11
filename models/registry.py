"""
Model Registry
==============
Centralized model registration and retrieval system.
"""

from typing import Dict, Type, List
from .base import BaseFuseTrainer


class ModelRegistry:
    """
    Registry for managing and accessing model classes.
    
    Example:
        @ModelRegistry.register('mymodel')
        class MyModel(BaseFuseTrainer):
            def __init__(self, hparams):
                super().__init__(hparams)
                # ...
    """
    _models: Dict[str, Type[BaseFuseTrainer]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator for registering a model class.
        
        Args:
            name: Model name identifier
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register('drfuse')
            class DrFuse(BaseFuseTrainer):
                pass
        """
        def wrapper(model_cls: Type[BaseFuseTrainer]) -> Type[BaseFuseTrainer]:
            cls._models[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get_model_cls(cls, name: str) -> Type[BaseFuseTrainer]:
        """
        Get registered model class by name.
        
        Args:
            name: Model name
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model not found in registry
        """
        if name not in cls._models:
            available = ', '.join(cls.list_models())
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        return cls._models[name]

    @classmethod
    def get_model(cls, name: str, hparams: dict) -> BaseFuseTrainer:
        """
        Get instantiated model by name.
        
        Args:
            name: Model name
            hparams: Model hyperparameters
            
        Returns:
            Instantiated model
        """
        model_cls = cls.get_model_cls(name)
        return model_cls(hparams)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        Get list of all registered model names.
        
        Returns:
            List of model names
        """
        return sorted(cls._models.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            name: Model name
            
        Returns:
            True if model is registered
        """
        return name in cls._models
