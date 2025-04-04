from doexpy.env.discrete_env import DiscreteEnv
# Removed direct CLIP imports, will use embedder object
from typing import List, Tuple, Union
import torch
from torch import nn
# Import the base embedder class for type hinting
from experiments.llm.components.embedder import BaseEmbedder
# Import PIL Image type hint
from PIL.Image import Image as PILImage
import os
import hashlib
import pickle
import numpy as np
import torch


class LLMGrid(DiscreteEnv):
    def __init__(
        self,
        list_of_text_tokens: List[str],
        embedder: BaseEmbedder, # Accept an embedder instance
        base_prompt: str = '',
        verbose: bool = False,
        include_base_prompt_in_first_tokens: bool = True,
    ):
        self.verbose = verbose
        self.constrained = False
        super().__init__(init_state=0)

        self.embedder = embedder # Store the embedder instance
        self.device = self.embedder.device # Get device from embedder
        # No need for separate processor/tokenizer storage if accessed via embedder
        # self.cache_dir = self.embedder.cache_dir # Can get from embedder if needed

        self.include_base_prompt_in_first_tokens = include_base_prompt_in_first_tokens
        self.base_prompt = base_prompt # Always store the actual base_prompt

        # Process token lists based on configuration
        if include_base_prompt_in_first_tokens and base_prompt:
            # Add base prompt to first token list only
            first_tokens = [f"{base_prompt}, {t}" if t != ' ' else f'{base_prompt}' for t in list_of_text_tokens[0]]
            list_of_text_tokens = [first_tokens] + [[f"{t}" for t in token_list] for token_list in list_of_text_tokens[1:]]
        else:
            list_of_text_tokens = [[f"{t}" for t in token_list] for token_list in list_of_text_tokens]

        self.max_episode_length = len(list_of_text_tokens)
        # Setup tokens dictionary
        self.tokens = {}
        index = 1
        self.tokens[' '] = [i for i in range(self.max_episode_length)]
        self.unique_elements = [' ']
        for order, list in enumerate(list_of_text_tokens):
            for token in list:
                if token not in self.tokens:
                    self.unique_elements.append(token)
                    self.tokens[token] = [order]
                    index += 1
                else:
                    self.tokens[token] += [order]

        total_tokens = len(self.unique_elements)

        self.states_num = self.max_episode_length
        self.actions_num = total_tokens
        self.h = 0
        self.action_space_pre_embedding = torch.arange(self.actions_num, dtype=torch.float64).to(self.device).reshape(-1, 1)
        self.emiss_num = self.actions_num
        self.transition_matrix = None
        # Pass the embedder instance to generate_emissions
        self.emissions = generate_emissions(self.unique_elements, self.embedder, self.verbose)
        self.action_space = self.emissions
        self.visitations = torch.zeros(self.states_num, self.actions_num, dtype=torch.float64).to(self.device)


    def get_dim(self):
        # Get dimension from the embedder
        return self.embedder.get_embedding_dim()

    def get_states_num(self):
        return self.states_num

    def next(self, state, action):
        return state + 1

    def convert(self, state):
        pass

    def available_actions(self, state):
        actions = []
        actions.append(0)
        for i in range(1, self.actions_num):
            if self.is_valid_action(i, state):
                actions.append(i)
        return actions

    def step(self, action):
        self.visitations[self.state, action] += 1
        self.state = self.next(self.state, action)
        return self.state

    def is_valid_action(self, action, state) -> bool:
        if action == 0 and state > 0:
            return True
        else:
            if state in self.tokens[self.unique_elements[action]]:
                return True
            else:
                return False

    def p_next(self, state, action):
        probs = {min(state + 1, self.max_episode_length - 1): 1}
        return probs

    def get_transition_matrix(self) -> torch.Tensor:
        if self.transition_matrix is not None:
            return self.transition_matrix

        P = torch.zeros(size=(self.states_num, self.actions_num, self.states_num), dtype=torch.float64)
        for s in range(self.states_num):
            for a in range(self.actions_num):
                if self.is_valid_action(a, s):
                    probs = self.p_next(s, a)
                    for s_state in probs.keys():
                        P[s, a, s_state] = probs[s_state]
        self.transition_matrix = P
        return P

    def reset(self) -> None:
        self.state = self.init_state
        self.h = 0

# Removed CLIPEmbedder class definition (moved to components/embedder.py)


class VisionLanguageScorer(nn.Module):
    """Base class for Vision-Language scoring models"""
    def __init__(self, embedder: BaseEmbedder):
        super().__init__()
        self.embedder = embedder # Store the embedder instance

    def score_prompt(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores a text prompt. Returns score and embedding."""
        raise NotImplementedError

    def score_embedding(self, x_embedding: torch.Tensor) -> torch.Tensor:
        """Scores a pre-computed embedding."""
        raise NotImplementedError


class DotProductModel(VisionLanguageScorer):
    """Scores prompts based on the dot product of their embedding with a weight vector."""
    def __init__(self, embedder: BaseEmbedder, weight: torch.Tensor, bias: torch.Tensor = None):
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


def generate_emissions(unique_elements: List[str], embedder: BaseEmbedder, verbose: bool = True) -> torch.Tensor:
    """Generate emissions (embeddings) for a list of unique elements using the provided embedder.

    Args:
        unique_elements: List of text tokens to generate emissions for.
        embedder: An instance of BaseEmbedder (e.g., CLIPEmbedder).
        verbose: Whether to print progress messages.

    Returns:
        torch.Tensor: Matrix of emissions [num_elements, embedding_dim].
    """
    cache_dir = embedder.cache_dir # Get cache dir from embedder
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a cache key based on elements and embedder configuration
    hasher = hashlib.sha256()
    hasher.update(str(len(unique_elements)).encode())
    hasher.update(embedder.get_config_hash().encode()) # Use embedder's config hash
    # Consider adding base_prompt if it influences token text (already handled in LLMGrid init?)
    # hasher.update(base_prompt.encode())
    for elem in unique_elements:
        hasher.update(elem.encode())
    cache_id = hasher.hexdigest()
    cache_path = os.path.join(cache_dir, f"emissions_{cache_id}.pkl")

    if os.path.exists(cache_path):
        if verbose:
            print("Loading emissions from cache")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, RuntimeError):
            if verbose:
                print("Cache file corrupted, regenerating")
            os.remove(cache_path)

    print("PREPROCESS: Generating emissions...")
    emissions = []
    total_elements = len(unique_elements)
    print_interval = 10  # Print progress every 10 iterations

    for i, text in enumerate(unique_elements):
        embed_text = text  # Always embed the text as is
        if verbose:
            # Verbose printing can be helpful for debugging token content
            if verbose:
                print(f"Embedding text: {text}")

        # Use the embedder's method
        feat = embedder.embed_text(text)
        emissions.append(feat)

        # Print progress
        if verbose and ((i + 1) % print_interval == 0 or (i + 1) == total_elements):
            print(f"Generated emission {i + 1}/{total_elements}")

    print("Done generating emissions.")
    emissions = torch.vstack(emissions)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(emissions, f)
    except Exception as e:
        if verbose:
            print(f"Failed to cache emissions: {e}")

    return emissions


def load_aesthetics_embedding(weights_path='vit_14_weights.pth', device=None):
    """Load aesthetics model weights.

    Note: These weights are specific to CLIP ViT-L/14. Using them with other
          embedders (like SigLIP or different CLIP models) will likely yield
          meaningless results due to dimension and embedding space mismatch.

    Args:
        weights_path: Path to aesthetics weights file (e.g., vit_l_14_weights.pth).
        device: Target device ('cuda', 'cpu', or None for auto-detect).

    Returns:
        torch.Tensor: Weight tensor on the specified device.
    """
    target_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(weights_path):
         # Try common locations if just filename is given
         potential_paths = [
             weights_path,
             os.path.join(os.path.dirname(__file__), weights_path), # Relative to this file
             os.path.join(os.path.expanduser("~/.cache"), weights_path) # A common cache spot
         ]
         found = False
         for p in potential_paths:
             if os.path.exists(p):
                 weights_path = p
                 found = True
                 break
         if not found:
            raise FileNotFoundError(f"Could not find aesthetics weights file: {weights_path} in likely locations.")

    try:
        # Aesthetics models usually don't have a bias term saved this way
        state = torch.load(weights_path, map_location=target_device)
        # Check common keys for the weight tensor
        if 'weight' in state:
            weight = state['weight']
        elif 'linear.weight' in state: # Another common pattern
            weight = state['linear.weight']
        else:
            # If it's just a tensor saved directly
            if isinstance(state, torch.Tensor):
                 weight = state
            else:
                 raise KeyError("Could not find weight tensor in the aesthetics state dictionary.")

        # Bias is typically not included or handled differently for aesthetics scores
        return weight.to(target_device).double() # Return only weight, ensure dtype
    except Exception as e:
        print(f"Error loading aesthetics weights from {weights_path}: {e}")
        raise


# Removed setup_clip_model function (handled by embedder factory)


def create_prompt_from_tokens(tokens: List[str], base_prompt: str = '') -> str:
    """Create a prompt from a list of tokens

    Args:
        tokens: List of text tokens
        base_prompt: Optional base prompt to prepend

    Returns:
        Formatted prompt string with commas
    """
    # Filter out empty tokens and strip whitespace
    valid_tokens = [str(token).strip() for token in tokens if str(token).strip()]

    if base_prompt:
        # If we have a base prompt, add the tokens after it with commas
        if valid_tokens:
            return base_prompt + ", " + ", ".join(valid_tokens)
        else:
            return base_prompt
    else:
        # If no base prompt, just join the tokens with commas
        return ", ".join(valid_tokens)


def create_prompt(actions: List[int], env: LLMGrid) -> str:
    """Create prompt string from a sequence of action indices.

    Args:
        actions: List of action indices.
        env: The LLMGrid environment instance.

    Returns:
        Formatted prompt string.
    """
    # Get token strings corresponding to action indices
    # Handle potential index errors if action is out of bounds
    tokens = []
    for action in actions:
        action_idx = int(action)
        if 0 <= action_idx < len(env.unique_elements):
            tokens.append(env.unique_elements[action_idx])
        else:
            print(f"Warning: Action index {action_idx} out of bounds for unique_elements (size {len(env.unique_elements)}). Skipping.")
            # Decide how to handle invalid actions: skip, raise error, use placeholder?
            # tokens.append("[INVALID_ACTION]") # Option: Placeholder

    # Base prompt handling depends on env configuration
    if env.include_base_prompt_in_first_tokens:
        # The base prompt is assumed to be part of the first token(s) already
        # (as handled in LLMGrid.__init__). We just join the retrieved tokens.
        # We need to filter out the placeholder ' ' token if it's the first action.
        if tokens and tokens[0] == ' ':
             # If the first action is the space placeholder, don't add extra commas
             effective_tokens = tokens[1:]
             # The first real token might already contain the base prompt.
             return create_prompt_from_tokens(effective_tokens, base_prompt='')
        else:
             return create_prompt_from_tokens(tokens, base_prompt='')
    else:
        # Prepend the base prompt if it's not included in the tokens
        return create_prompt_from_tokens(tokens, base_prompt=env.base_prompt)


def get_scorer_model(model_name: str, env: LLMGrid, embedder: BaseEmbedder) -> VisionLanguageScorer:
    """Initialize the scoring model based on the specified type.

    Args:
        model_name: Scorer type ('japanese-text', 'japanese-image',
                    'aesthetics', 'random_combination').
        env: The LLMGrid environment instance (used for emissions).
        embedder: The embedder instance (used for embedding prompts/images
                  and determining dimensions/device).

    Returns:
        An instance of VisionLanguageScorer (e.g., DotProductModel).
    """
    emissions_env = env.emissions # Embeddings of the unique elements/actions

    if model_name == 'japanese-text':
        # Use the embedder to get the weight vector from text
        prompt = "An image with clear observable japanese influence, japanese history, japanese traditions or japanese symbols"
        weight_vector = embedder.embed_text(prompt)
        return DotProductModel(embedder, weight_vector).eval()

    elif model_name == 'japanese-image':
        # Use the embedder to get the weight vector from an image
        from PIL import Image
        import os

        image_path = 'japan.jpg' # Assumed relative to execution or in PYTHONPATH
        potential_paths = [image_path, os.path.join(os.path.dirname(__file__), image_path)]
        found_path = None
        for p in potential_paths:
            if os.path.exists(p):
                found_path = p
                break
        if not found_path:
             raise FileNotFoundError(f"Image file not found: {image_path} in likely locations.")

        image = Image.open(found_path).convert("RGB") # Ensure RGB
        weight_vector = embedder.embed_image(image)
        return DotProductModel(embedder, weight_vector).eval()

    elif model_name == 'aesthetics':
        # Load pre-computed aesthetics weights (CLIP ViT-L/14 specific!)
        print("Warning: Using 'aesthetics' scorer assumes a CLIP ViT-L/14 compatible embedder.")
        try:
            # Pass device from embedder
            aes_weight = load_aesthetics_embedding(device=embedder.device)
            # Bias is typically not used or is 0 for these models
            return DotProductModel(embedder, aes_weight, bias=None).eval()
        except FileNotFoundError as e:
            print(f"Error: {e}. Aesthetics scorer requires weights file.")
            raise
        except ValueError as e:
             # Catch dimension mismatch if DotProductModel raises it
             print(f"Error initializing aesthetics scorer: {e}")
             print("Ensure the embedder's dimension matches the aesthetics weights.")
             raise

    elif model_name == 'random_combination':
        # Create a weight vector as a random combination of action embeddings
        rng = np.random.RandomState(42)
        device = embedder.device
        dtype = emissions_env.dtype # Use dtype from existing emissions

        num_actions = emissions_env.shape[0]
        k = min(25, num_actions) # Ensure k is not larger than the number of actions
        if k == 0:
             raise ValueError("Cannot create random combination scorer with zero actions/emissions.")

        selected_indices = rng.choice(num_actions, k, replace=False)

        # Create coefficients on the correct device and dtype
        random_coeffs = torch.zeros(num_actions, device=device, dtype=dtype)
        selected_coeffs = 2 * torch.rand(k, device=device, dtype=dtype) - 1 # Uniform in [-1, 1]
        random_coeffs[selected_indices] = selected_coeffs

        # Calculate the weighted combination of emissions
        # random_coeffs shape: [num_actions]
        # emissions_env shape: [num_actions, embedding_dim]
        # Result shape: [embedding_dim] -> view as [1, embedding_dim]
        weight_vector = torch.matmul(random_coeffs.unsqueeze(0), emissions_env) # [1, num_actions] @ [num_actions, dim] -> [1, dim]

        return DotProductModel(embedder, weight_vector).eval()

    raise ValueError(f"Unknown scorer model name: {model_name}")


def make_theta_star(env: LLMGrid, scorer_model: VisionLanguageScorer, verbose: bool = False):
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
    def theta_star(actions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
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
