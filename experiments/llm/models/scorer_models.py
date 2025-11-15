import os
import hashlib
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

# Import for type hinting without circular dependency
if TYPE_CHECKING:
    from doexpy.env.llm import LLMGrid
    from components.embedder import BaseEmbedder

# Import for sunny-image model type in get_scorer_model
from image_generator import StableDiffusionGenerator, DEFAULT_CONFIG
from PIL import Image # Changed import for fromarray

# Import create_prompt for make_theta_star
from doexpy.env.llm import create_prompt


class VisionLanguageScorer(nn.Module):
    """Base class for Vision-Language scoring models"""
    def __init__(self, embedder: 'BaseEmbedder'):
        super().__init__()
        self.embedder = embedder # Store the embedder instance

    @abstractmethod
    def score_prompt(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores a text prompt. Returns score and embedding."""
        raise NotImplementedError

    @abstractmethod
    def score_embedding(self, x_embedding: torch.Tensor) -> torch.Tensor:
        """Scores a pre-computed embedding."""
        raise NotImplementedError


class DotProductModel(VisionLanguageScorer):
    """Scores prompts based on the dot product of their embedding with a weight vector."""
    def __init__(self, embedder: 'BaseEmbedder', weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__(embedder)
        expected_dim = self.embedder.get_embedding_dim()

        # Validate and standardize weight shape
        if weight.dim() == 1:
            weight = weight.view(1, -1) # Convert 1D to [1, dim]
        elif weight.shape[0] != 1 and weight.shape[1] == 1:
            weight = weight.T # Convert [dim, 1] to [1, dim]
        elif weight.shape[0] != 1:
             raise ValueError(f"Weight must be 1D or have one dimension of size 1, got shape {weight.shape}")

        if weight.shape[1] != expected_dim:
            raise ValueError(f"Weight dimension ({weight.shape[1]}) does not match embedder dimension ({expected_dim})")

        self.weight = weight.to(self.embedder.device).double() # Ensure correct device and dtype
        self.bias = bias.to(self.embedder.device).double() if bias is not None else None
        # Removed erroneous else block here
    def score_embedding(self, x_embedding: torch.Tensor) -> torch.Tensor:
        """Score a pre-computed embedding directly."""
        # Ensure input has correct shape [batch_size, embedding_dim]
        if x_embedding.dim() == 1:
            x_embedding = x_embedding.view(1, -1)

        # Verify shapes are compatible (already checked in __init__, but good practice)
        if x_embedding.shape[1] != self.weight.shape[1]:
            raise ValueError(f"Input embedding dimension ({x_embedding.shape[1]}) doesn't match weight dimension ({self.weight.shape[1]})")

        # Ensure correct device and dtype
        x_embedding = x_embedding.to(self.embedder.device).double()

        # Simple dot product
        score = torch.mm(x_embedding, self.weight.T)

        if self.bias is not None:
            score += self.bias

        return score

    def score_prompt(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a text prompt by first embedding then scoring."""
        x_embedding = self.embedder.embed_text(x)
        score = self.score_embedding(x_embedding)
        return score, x_embedding


def get_scorer_model(model_name: str, env: 'LLMGrid', embedder: 'BaseEmbedder') -> VisionLanguageScorer:
    """Initialize the scoring model based on the specified type.

    Args:
        model_name: Scorer type (e.g., 'sunny', 'medieval').
        env: The LLMGrid environment instance (used for emissions).
        embedder: The embedder instance (used for embedding prompts/images
                  and determining dimensions/device).

    Returns:
        An instance of VisionLanguageScorer (e.g., DotProductModel).
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Handle special 'sunny-image' model first
    if model_name == 'sunny-image':
        print(f"Initializing ground truth scorer model: {model_name} (average of image embeddings from sunny.txt)")
        # Construct path to sunny.txt
        sentences_file_path = os.path.join(current_script_dir, 'sunny.txt')

        if not os.path.exists(sentences_file_path):
            raise FileNotFoundError(f"Sentences file for sunny-image model not found at: {sentences_file_path}")

        # --- Caching Logic for sunny-image model weights ---
        cache_dir = embedder.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Create a hash for image generator configuration used for this GT model
        image_gen_config_hasher = hashlib.sha256()
        image_gen_config_hasher.update(DEFAULT_CONFIG['stable_diffusion_id'].encode())
        image_gen_config_hasher.update(str(DEFAULT_CONFIG['image_size']).encode())
        image_gen_config_hasher.update(str(DEFAULT_CONFIG['num_inference_steps']).encode())
        image_gen_config_hasher.update(str(42).encode()) # Fixed seed (42) for GT model reproducibility
        image_gen_config_hash = image_gen_config_hasher.hexdigest()

        # Create a hash for the sentences file content
        sentences_file_content_hasher = hashlib.sha256()
        try:
            with open(sentences_file_path, 'rb') as f_sentences: # Read as binary for hashing
                buf = f_sentences.read(65536)
                while len(buf) > 0:
                    sentences_file_content_hasher.update(buf)
                    buf = f_sentences.read(65536)
            sentences_file_hash = sentences_file_content_hasher.hexdigest()
        except Exception as e:
            raise RuntimeError(f"Failed to hash sentences file {sentences_file_path}: {e}")

        cache_key_parts = [
            model_name, # "sunny-image"
            sentences_file_hash,
            embedder.get_config_hash(),
            image_gen_config_hash
        ]
        cache_id = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()
        weight_vector_cache_path = os.path.join(cache_dir, f"gt_model_weights_{cache_id}.pt")

        if os.path.exists(weight_vector_cache_path):
            print(f"  Loading cached ground truth model weights for '{model_name}' from: {weight_vector_cache_path}")
            try:
                weight_vector = torch.load(weight_vector_cache_path, map_location=embedder.device)
                weight_vector = weight_vector.to(device=embedder.device, dtype=torch.double) # Ensure correct device and dtype
                print(f"  Successfully loaded cached weights for '{model_name}'.")
                return DotProductModel(embedder, weight_vector).eval()
            except Exception as e:
                print(f"  Warning: Failed to load cached weights for '{model_name}' from {weight_vector_cache_path}: {e}. Regenerating.")
                try:
                    os.remove(weight_vector_cache_path)
                except OSError as remove_err:
                    print(f"  Warning: Could not remove corrupted cache file {weight_vector_cache_path}: {remove_err}")
        
        print(f"  No valid cache found for '{model_name}' weights. Generating...")
        # Instantiate StableDiffusionGenerator with a fixed seed for reproducibility
        try:
            image_generator = StableDiffusionGenerator(
                stable_diffusion_id=DEFAULT_CONFIG['stable_diffusion_id'],
                MODELS_CACHE_DIR=DEFAULT_CONFIG['MODELS_CACHE_DIR'],
                image_size=DEFAULT_CONFIG['image_size'], 
                num_inference_steps=DEFAULT_CONFIG['num_inference_steps'],
                seed=42 # Fixed seed for GT model reproducibility
            )
            print(f"  Instantiated StableDiffusionGenerator for {model_name} GT model creation.")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate StableDiffusionGenerator for {model_name}: {e}")

        image_embeddings_list = []
        with open(sentences_file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"  Generating images and embeddings for {len(sentences)} sentences from {os.path.basename(sentences_file_path)}...") # Use basename for print
        for i, sentence in enumerate(sentences):
            try:
                generated_image_np, img_embedding_tensor = image_generator.sample(prompt=sentence, embedder=embedder)
                generated_image_pil = Image.fromarray(generated_image_np) # Use Image.fromarray
                img_embedding = embedder.embed_image(generated_image_pil)
                image_embeddings_list.append(img_embedding)
                if env.verbose or (i + 1) % 10 == 0 or i == len(sentences) - 1:
                    print(f"    Processed sentence {i+1}/{len(sentences)} for {model_name} GT.")
            except Exception as e:
                print(f"    Error processing sentence '{sentence}' for {model_name} GT: {e}. Crashing.")
                raise # Re-raise the caught exception to stop execution
        
        if not image_embeddings_list:
            raise ValueError(f"No valid image embeddings generated for '{model_name}' model from '{sentences_file_path}'.")

        stacked_image_embeddings = torch.cat(image_embeddings_list, dim=0)
        weight_vector = stacked_image_embeddings.mean(dim=0).to(device=embedder.device, dtype=torch.double)
        
        print(f"  Finished creating weight vector for {model_name} from {len(image_embeddings_list)} image embeddings.")

        # Save the newly computed weight_vector to cache
        try:
            torch.save(weight_vector, weight_vector_cache_path)
            print(f"  Saved ground truth model weights for '{model_name}' to cache: {weight_vector_cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to save ground truth model weights for '{model_name}' to cache {weight_vector_cache_path}: {e}")

        return DotProductModel(embedder, weight_vector).eval()

    # Generic handler for any model defined by a .txt file
    sentences_file_path = os.path.join(current_script_dir, f'{model_name}.txt')
    
    if os.path.exists(sentences_file_path):
        normalized_embeddings = []
        # Ensure to use utf-8 encoding for reading text files
        with open(sentences_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence: # Process non-empty lines
                    embedding = embedder.embed_text(sentence) # Expected to be [1, dim]
                    # The embedder's `embed_text` method handles normalization
                    # if its `normalize` attribute is True.
                    normalized_embeddings.append(embedding)

        if not normalized_embeddings:
            # This case handles empty file or file with only empty lines/problematic embeddings
            raise ValueError(f"No valid sentences found in '{sentences_file_path}' to create '{model_name}' model.")

        # Stack embeddings into a 2D tensor [num_sentences, embedding_dim]
        stacked_embeddings = torch.stack(normalized_embeddings)
        
        # Compute the mean embedding. Ensure it's on the correct device and dtype.
        # DotProductModel expects weights to be torch.double.
        weight_vector = stacked_embeddings.mean(dim=0).to(device=embedder.device, dtype=torch.double)
        
        return DotProductModel(embedder, weight_vector).eval()

    # If neither of the above conditions were met, the model is unknown.
    raise ValueError(f"Unknown scorer model name: '{model_name}'. No special handler exists and the corresponding file '{os.path.basename(sentences_file_path)}' was not found in the models directory.")


def make_theta_star(env: 'LLMGrid', scorer_model: VisionLanguageScorer, verbose: bool = False):
    """
    Creates the ground truth scoring function theta_star.

    Args:
        env: The LLMGrid environment instance.
        scorer_model: The initialized scoring model (e.g., DotProductModel).
        verbose: If True, print the generated prompt before scoring.

    Returns:
        A function `theta_star(actions)` that takes a list of action indices
        and returns the score and the corresponding embedding.
    """
    def theta_star(actions: list[int]) -> Tuple[torch.Tensor, torch.Tensor]: # Use list[int] for type hint
        """The ground truth scoring function."""
        if not actions:
            # Handle empty action list if necessary, maybe return zero score and zero embedding?
            print("Warning: theta_star called with empty action list.")
            emb_dim = env.get_dim()
            zero_score = torch.tensor([[0.0]], device=env.device, dtype=torch.double)
            zero_embedding = torch.zeros((1, emb_dim), device=env.device, dtype=torch.double)
            return zero_score, zero_embedding
            # Alternatively, raise ValueError("Action list cannot be empty.")

        # Create the full prompt string from actions
        prompt = create_prompt(actions, env)

        if verbose:
            print(f"Theta* scoring prompt: '{prompt}'")

        # Use the scorer model's method to get score and embedding
        score, embedding = scorer_model.score_prompt(prompt)
        return score, embedding

    return theta_star
