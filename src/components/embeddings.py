import os
import sys
import importlib.util
from typing import List, Union, Dict, Any
import numpy as np

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Import settings
from config.settings import EMBEDDING_MODELS, ACTIVE_EMBEDDING_MODEL


class EmbeddingManager:
    """
    Manages embedding generation using various providers and models.

    This class provides a unified interface for generating embeddings
    regardless of the underlying model or provider.
    """

    def __init__(self, model_key: str = None):
        """
        Initialize the embedding manager.

        Args:
            model_key: Key for the embedding model to use from settings.
                       If None, uses the ACTIVE_EMBEDDING_MODEL from settings.
        """
        if model_key is None:
            model_key = ACTIVE_EMBEDDING_MODEL

        if model_key not in EMBEDDING_MODELS:
            raise ValueError(
                f"Model key '{model_key}' not found in settings. Available models: {list(EMBEDDING_MODELS.keys())}"
            )

        self.model_config = EMBEDDING_MODELS[model_key]
        self.model_key = model_key
        self.provider = self.model_config["provider"]
        self.model_name = self.model_config["name"]
        self.dimensions = self.model_config["dimensions"]

        # Initialize the appropriate model
        if self.provider == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "The sentence-transformers package is required for this embedding provider. "
                    "Please install it with: pip install sentence-transformers"
                )
        elif self.provider == "openai":
            try:
                import openai

                # Just store configuration, API calls made during embedding
                self.model = None
            except ImportError:
                raise ImportError(
                    "The openai package is required for this embedding provider. "
                    "Please install it with: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given texts.

        Args:
            texts: A string or list of strings to generate embeddings for

        Returns:
            Numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings based on provider
        if self.provider == "sentence_transformers":
            return self.model.encode(texts)
        elif self.provider == "openai":
            import openai
            import numpy as np

            # Set up OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            client = openai.OpenAI(api_key=api_key)

            # Get embeddings from OpenAI API
            response = client.embeddings.create(input=texts, model=self.model_name)

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of the embeddings"""
        return self.dimensions

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the current model"""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "provider": self.provider,
            "dimensions": self.dimensions,
            "description": self.model_config.get(
                "description", "No description available"
            ),
        }

    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary of available models and their configurations"""
        return EMBEDDING_MODELS
