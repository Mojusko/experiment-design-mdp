from doexpy.env.discrete_env import DiscreteEnv
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from typing import List, Tuple, Union
import torch
from torch import nn
import os
import hashlib
import pickle
import numpy as np
import torch # Added for prior generation
from experiments.llm.image_generator import StableDiffusionGenerator, DEFAULT_CONFIG as IMAGE_GEN_DEFAULT_CONFIG # Added for prior generation


class LLMGrid(DiscreteEnv):
    def __init__(
        self,
        list_of_text_tokens: List[str],
        model: CLIPModel,
        processor: CLIPProcessor,
        tokenizer: CLIPTokenizer,
        cache_dir: str,
        normalize_embedder: bool,
        base_prompt: str = '',
        verbose: bool = False,
        include_base_prompt_in_first_tokens: bool = True,
        prior_prompt: str = None, # Added parameter for prior generation
        image_gen_config: dict = None, # Added parameter for image gen settings
    ):
        self.verbose = verbose
        self.constrained = False
        super().__init__(init_state=0)
        self._prior_vector = None # Initialize prior vector

        self.device = next(model.parameters()).device
        self.embedder = CLIPEmbedder(tokenizer, model, normalize=normalize_embedder)
        self._processor = processor
        self._tokenizer = tokenizer

        self.cache_dir = cache_dir

        self.include_base_prompt_in_first_tokens = include_base_prompt_in_first_tokens
        self.base_prompt = base_prompt  # Always store the actual base_prompt

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
        self.emissions = generate_emissions(self.unique_elements, self.embedder, self.cache_dir, self.verbose)
        self.action_space = self.emissions
        self.visitations = torch.zeros(self.states_num, self.actions_num, dtype=torch.float64).to(self.device)

        # Generate prior vector from image if prior_prompt is provided
        if prior_prompt:
            if verbose:
                print(f"LLMGrid: Generating prior image embedding for C-optimal design using prompt: '{prior_prompt}'")

            # Determine image generation settings based on debug mode in config
            gen_config = image_gen_config or {}
            is_debug = gen_config.get('debug_mode', False)

            if is_debug:
                if verbose:
                    print("LLMGrid: Using DEBUG settings for prior image generation.")
                num_steps = gen_config.get('num_inference_steps', 20) # Default debug steps
                img_size = gen_config.get('image_size', 32) # Default debug size
            else:
                if verbose:
                    print("LLMGrid: Using DEFAULT settings for prior image generation.")
                num_steps = IMAGE_GEN_DEFAULT_CONFIG["num_inference_steps"]
                img_size = IMAGE_GEN_DEFAULT_CONFIG["image_size"]

            # Use determined settings for the generator
            prior_generator = StableDiffusionGenerator(
                stable_diffusion_id=IMAGE_GEN_DEFAULT_CONFIG["stable_diffusion_id"],
                num_inference_steps=num_steps,
                guidance_scale=IMAGE_GEN_DEFAULT_CONFIG["guidance_base"], # Keep default guidance
                image_size=img_size,
                # Seed is derived from the prompt internally by the generator
                MODELS_CACHE_DIR=self.cache_dir # Use environment's cache dir
            )
            _, prior_image_embedding = prior_generator.sample(prior_prompt)

            # Normalize the prior embedding (L2 normalization)
            prior_image_embedding_norm = prior_image_embedding / torch.norm(prior_image_embedding, p=2)

            # Store the normalized prior embedding in the environment
            # Ensure it's on the correct device and dtype
            self._prior_vector = prior_image_embedding_norm.to(self.device).to(self.emissions.dtype).unsqueeze(0) # Ensure shape [1, dim]
            if verbose:
                print(f"LLMGrid: Stored prior vector with shape: {self._prior_vector.shape}")

    def get_prior_vector(self):
        """Returns the generated prior vector, if available."""
        return self._prior_vector

    def get_dim(self):
        return 768

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


class CLIPEmbedder:
    """Embed text using CLIP model

    Args:
        text: Text to embed
        normalize: Whether to L2 normalize the embedding

    Returns:
        Text embedding
    """
    def __init__(self, tokenizer, model, normalize: bool = True):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device  # Track model device
        self.normalize = normalize

    def embed_text(self, text: str) -> torch.Tensor:
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        embedding = self.model.get_text_features(**text_input).detach().double()

        if self.normalize:
            embedding = embedding / torch.norm(embedding, p=2)

        return embedding.view(1, -1)


class CLIPScorer(nn.Module):
    """Base class for CLIP-based scoring models"""
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def score_prompt(self, x):
        raise NotImplementedError


class DotProductModel(CLIPScorer):
    def __init__(self, embedder, weight, bias=None):
        super().__init__(embedder)
        # Standardize weight to always be a 2D tensor with shape [1, embedding_dim]
        if weight.dim() == 1:
            self.weight = weight.view(1, -1).to(embedder.device)
        else:
            # If it's already 2D, ensure it's [1, embedding_dim] or [embedding_dim, 1]
            if weight.shape[0] == 1 or weight.shape[1] == 1:
                # Make sure it's [1, embedding_dim]
                if weight.shape[1] == 1:
                    self.weight = weight.T.to(embedder.device)
                else:
                    self.weight = weight.to(embedder.device)
            else:
                raise ValueError(f"Weight must be 1D or have one dimension of size 1, got shape {weight.shape}")

        self.bias = bias.to(embedder.device) if bias is not None else None

    def score_embedding(self, x_clip_embedding):
        """Score a CLIP embedding directly"""
        # Ensure input has correct shape [batch_size, embedding_dim]
        if x_clip_embedding.dim() == 1:
            x_clip_embedding = x_clip_embedding.view(1, -1)

        # Verify shapes are compatible
        if x_clip_embedding.shape[1] != self.weight.shape[1]:
            raise ValueError(f"Embedding dimension {x_clip_embedding.shape[1]} doesn't match weight dimension {self.weight.shape[1]}")

        # Simple dot product
        score = torch.mm(x_clip_embedding, self.weight.T)

        if self.bias is not None:
            score += self.bias

        return score

    def score_prompt(self, x):
        """Score a text prompt by first embedding then scoring"""
        x_clip_embedding = self.embedder.embed_text(x)
        score = self.score_embedding(x_clip_embedding)
        return score, x_clip_embedding


def generate_emissions(unique_elements, embedder, cache_dir, verbose=True):
    """Generate emissions for a list of unique elements

    Args:
        unique_elements: List of text tokens to generate emissions for
        embedder: CLIPEmbedder instance with normalize attribute
        cache_dir: Directory for caching emissions
        verbose: Whether to print progress messages

    Returns:
        torch.Tensor: Matrix of emissions
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Include normalization in cache key
    hasher = hashlib.sha256()
    hasher.update(str(len(unique_elements)).encode())
    hasher.update(str(getattr(embedder, 'normalize', False)).encode())
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
            print(f"Embedding text: {embed_text}") # Keep verbose detail if needed

        feat = embedder.embed_text(embed_text)
        emissions.append(feat)

        # Print progress every `print_interval` iterations or on the last iteration
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


def load_aesthetics_embedding(weights_path='vit_14_weights.pth'):
    """Load aesthetics model weights and bias

    Args:
        weights_path: Path to aesthetics weights file

    Returns:
        tuple: (weight tensor, bias tensor) both on appropriate device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        state = torch.load(weights_path, map_location=device)
        weight = state['weight'].to(device).double()
        return weight, None
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")


def setup_clip_model(cache_dir):
    """Initialize shared CLIP model"""
    model_id = "openai/clip-vit-large-patch14"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    return model, processor, tokenizer


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


def create_prompt(actions: List[int], env) -> str:
    """Create prompt from action sequence

    Args:
        actions: List of action indices
        env: Environment with unique_elements, base_prompt, and include_base_prompt_in_first_tokens attributes

    Returns:
        Formatted prompt string with hashtags
    """
    tokens = [env.unique_elements[int(action)] for action in actions]
    if env.include_base_prompt_in_first_tokens:
        # Base prompt is already included in the first token, so don't add it again
        return create_prompt_from_tokens(tokens, base_prompt='')
    else:
        return create_prompt_from_tokens(tokens, base_prompt=env.base_prompt)


def get_scorer_model(model_name: str, env, clip_model, clip_processor, cache_dir):
    """Initialize embedder and scoring model

    Args:
        model_name: Scorer type ('japanese-text', 'japanese-image', 'aesthetics', 'random_combination')
        env: Environment object
        clip_model: CLIP model, required for image-based scorers
        clip_processor: CLIP processor, required for aesthetics-image
        cache_dir: Cache directory for image scorers

    Returns:
        Scoring model instance
    """
    emissions_env = env.emissions
    scorer_embedder = env.embedder

    if model_name == 'japanese-text':
        # Use a specific text prompt for the scorer weight
        prompt = f"An image with clear observable japanese influence, japanese history, japanese traditions or japanese symbols"
        embedding = scorer_embedder.embed_text(prompt)
        return DotProductModel(scorer_embedder, embedding).eval()

    elif model_name == 'japanese-image':
        # Load and embed the japan.jpg image using CLIP
        from PIL import Image
        import os
    
        # Load the image from the llm directory
        image_path = 'japan.jpg'
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
        image = Image.open(image_path)
    
        # Process image with CLIP
        inputs = clip_processor(
            images=image, 
            return_tensors="pt"
        ).to(env.embedder.device)
    
        # Get CLIP image features
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs).detach().double()
            # L2 normalize the embedding
            image_embedding = image_embedding / torch.norm(image_embedding, p=2)
            # Make sure it's a 2D tensor with shape [1, embedding_dim]
            if image_embedding.dim() == 1:
                image_embedding = image_embedding.view(1, -1)
    
        return DotProductModel(scorer_embedder, image_embedding).eval()

    if model_name == 'aesthetics':
        aes_weight, aes_bias = load_aesthetics_embedding()
        return DotProductModel(scorer_embedder, aes_weight, bias=aes_bias).eval()

    if model_name == 'random_combination':
        rng = np.random.RandomState(42)
        device = emissions_env.device
        dtype = emissions_env.dtype

        k = 25
        selected_indices = rng.choice(emissions_env.shape[0], k, replace=False)

        random_coeffs = torch.zeros(emissions_env.shape[0], device=device, dtype=dtype)
        selected_coeffs = 2 * rng.rand(k) - 1  # Uniform in [-1, 1]
        random_coeffs[selected_indices] = torch.tensor(selected_coeffs, device=device, dtype=dtype)

        random_combination_vec = torch.mm(random_coeffs.view(1, -1), emissions_env)
        return DotProductModel(scorer_embedder, random_combination_vec).eval()

    raise ValueError(f"Unknown model_name: {model_name}")


def make_theta_star(env, scorer_model, verbose=False):
    def theta_star(actions: List[int]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(actions) > 0
        prompt = create_prompt(actions, env)

        if verbose:
            print(prompt)

        score, clip_embedding = scorer_model.score_prompt(prompt)
        return score, clip_embedding

    return theta_star
