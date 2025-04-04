import torch
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import os
import pickle
import hashlib
from omegaconf import DictConfig
from PIL.Image import Image as PILImage # Use specific import to avoid potential conflicts

# Define a base class for embedders using ABC
class BaseEmbedder(ABC):
    """Abstract Base Class for text and image embedders."""

    def __init__(self, model_id: str, cache_dir: str, normalize: bool, device: str = None):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.normalize = normalize
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._load_model() # Load model during initialization

    @abstractmethod
    def _load_model(self):
        """Load the specific model, processor, and tokenizer."""
        pass

    @property
    def device(self):
        """Return the device the model is on."""
        return self._device

    @property
    def model(self):
        """Return the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        return self._model

    @property
    def processor(self):
        """Return the loaded processor."""
        if self._processor is None:
            raise RuntimeError("Processor not loaded. Call _load_model() first.")
        return self._processor

    @property
    def tokenizer(self):
        """Return the loaded tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call _load_model() first.")
        return self._tokenizer

    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a string of text."""
        pass

    @abstractmethod
    def embed_image(self, image: PILImage) -> torch.Tensor:
        """Embed a PIL image."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        pass

    def get_config_hash(self) -> str:
        """Generate a hash based on relevant configuration for caching."""
        hasher = hashlib.sha256()
        hasher.update(self.__class__.__name__.encode())
        hasher.update(self.model_id.encode())
        hasher.update(str(self.normalize).encode())
        # Add other relevant config items if needed in subclasses
        return hasher.hexdigest()


# --- Concrete Implementation: CLIPEmbedder ---

class CLIPEmbedder(BaseEmbedder):
    """Embed text or images using a CLIP model."""

    def _load_model(self):
        """Load CLIP model, processor, and tokenizer."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = AutoModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
        self._model.eval() # Set model to evaluation mode

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        """Embed text using CLIP."""
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        # Use text_model explicitly if available, otherwise main model
        if hasattr(self.model, 'text_model'):
             embedding = self.model.text_model(**text_input)[1] # [1] for pooled output
             embedding = self.model.text_projection(embedding)
        else:
             embedding = self.model.get_text_features(**text_input)

        embedding = embedding.detach().double()

        if self.normalize:
            embedding = embedding / torch.norm(embedding, p=2, dim=-1, keepdim=True)

        return embedding.view(1, -1) # Ensure [1, dim] shape

    @torch.no_grad()
    def embed_image(self, image: PILImage) -> torch.Tensor:
        """Embed an image using CLIP."""
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # Use vision_model explicitly if available, otherwise main model
        if hasattr(self.model, 'vision_model'):
            embedding = self.model.vision_model(**inputs)[1] # [1] for pooled output
            embedding = self.model.visual_projection(embedding)
        else:
            embedding = self.model.get_image_features(**inputs)

        embedding = embedding.detach().double()

        if self.normalize:
            embedding = embedding / torch.norm(embedding, p=2, dim=-1, keepdim=True)

        return embedding.view(1, -1) # Ensure [1, dim] shape

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for CLIP."""
        # Attempt to get from config, fallback to common values
        if hasattr(self.model.config, 'projection_dim') and self.model.config.projection_dim:
             return self.model.config.projection_dim
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
             # Sometimes projection_dim is missing, use text hidden size as fallback
             return self.model.config.text_config.hidden_size
        else:
             # Fallback for older models or unexpected configs
             print("Warning: Could not reliably determine embedding dimension from model config. Assuming 768.")
             return 768 # Common for large models


# --- Factory Function ---

def create_embedder(embedder_cfg: DictConfig) -> BaseEmbedder:
    """
    Factory function to create an embedder instance based on configuration.

    Args:
        embedder_cfg: The OmegaConf DictConfig object for the embedder.
                      Expected to have '_target_' pointing to the embedder class
                      and other parameters like 'model_id', 'cache_dir', 'normalize'.

    Returns:
        An instance of a BaseEmbedder subclass.
    """
    # Simple factory: directly instantiate using Hydra's mechanism
    # This assumes embedder_cfg has a _target_ key pointing to the class
    # and the class __init__ matches the parameters in the config.
    print(f"--- Debug: Instantiating embedder with config ---")
    print(embedder_cfg)
    print(f"--- End Debug ---")
    try:
        # Use hydra.utils.instantiate if available, otherwise manual
        import hydra
        return hydra.utils.instantiate(embedder_cfg)
    except ImportError:
        print("Hydra not found, attempting manual instantiation.")
        target_class_str = embedder_cfg.get("_target_")
        if not target_class_str:
            raise ValueError("Embedder configuration must contain a '_target_' key.")

        # Basic manual instantiation (less robust than Hydra's)
        parts = target_class_str.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        try:
            module = __import__(module_name, fromlist=[class_name])
            EmbedderClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import embedder class {target_class_str}: {e}")

        # Prepare args, removing _target_
        args = {k: v for k, v in embedder_cfg.items() if k != '_target_'}
        return EmbedderClass(**args)
    except Exception as e:
        raise ValueError(f"Failed to instantiate embedder from config {embedder_cfg}: {e}")

