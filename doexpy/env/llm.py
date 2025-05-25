from doexpy.env.discrete_env import DiscreteEnv
# Removed direct CLIP imports, will use embedder object
from typing import List, Tuple, Union, TYPE_CHECKING
import torch
from torch import nn
# Import the base embedder class and specific implementations for type hinting/checking
from experiments.llm.components.embedder import BaseEmbedder # Removed CLIPEmbedder, BaseEmbedder is enough
# Import PIL Image type hint
from PIL.Image import Image as PILImage
import os
import hashlib
# import pickle # No longer used here
import numpy as np
import torch
import random
from collections import Counter


class LLMGrid(DiscreteEnv):
    EMPTY_ACTION_TOKEN = " " # Define the empty action token

    def __init__(
        self,
        list_of_text_tokens: List[str],
        embedder: BaseEmbedder, # Accept an embedder instance
        base_prompt: str = '',
        verbose: bool = False,
        include_base_prompt_in_first_tokens: bool = True,
        rng=None, # Add rng argument
        handle_duplicate_action: bool = True,
    ):
        self.verbose = verbose
        self.constrained = False
        super().__init__(init_state=0)

        self.embedder = embedder # Store the embedder instance
        self.rng = rng if rng is not None else np.random.RandomState() # Store rng
        self.handle_duplicate_action = handle_duplicate_action # Store the flag
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
        self.tokens[' '] = [i for i in range(1,self.max_episode_length)]
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
        emissions_raw = generate_emissions(self.unique_elements, self.embedder, self.verbose)
        # Explicitly ensure emissions are on the correct device after loading/generation
        self.emissions = emissions_raw.to(self.device) 
             
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
        # Only add actions if they are valid for the current state
        if self.is_valid_action(0, state):
             actions.append(0)
        for i in range(1, self.actions_num):
            if self.is_valid_action(i, state):
                actions.append(i)
        return actions

    def step(self, action):
        self.visitations[self.state, action] += 1
        self.state = self.next(self.state, action)
        return self.state

    def is_valid_action(self, action_id: int, current_depth_state: int) -> bool:
        """
        Checks if an action (token) is valid for the current depth in the prompt.
        - At depth 0, tokens must come from the first vocabulary list (bases.txt).
        - At depth > 0, tokens can come from any *other* vocabulary list.
        """
        token_str = self.unique_elements[action_id]
        # original_vocab_indices_for_token lists the indices of vocabulary files
        # (e.g., 0 for bases.txt, 1 for ambient.txt) where this token appears.
        original_vocab_indices_for_token = self.tokens[token_str]

        if current_depth_state == 0:
            # For the first token of the prompt (depth 0),
            # it must be from the first vocabulary list (index 0).
            return 0 in original_vocab_indices_for_token
        else:
            # For subsequent tokens (depth > 0),
            # it must be from any vocabulary list *other than* the first one.
            # This includes the special ' ' token, which is in self.tokens[' '] = [1, ..., H-1].
            return any(k_idx > 0 for k_idx in original_vocab_indices_for_token)

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

    def _handle_duplicate_actions(self, actions: List[int]) -> List[int]:
        """
        Processes a list of actions to handle duplicates.
        Duplicate non-zero actions are replaced with 0, keeping only one instance
        chosen uniformly at random from its occurrences.
        Uses self.rng for the random choice.
        """
        if not self.handle_duplicate_action:
            return actions # Return original actions if handling is disabled

        if not actions:
            return []

        # Ensure actions are integers for Counter and dictionary keys
        processed_actions = [int(act) for act in actions]
        
        counts = Counter(processed_actions)
        keep_map = {}

        for action_value, count in counts.items():
            if count > 1 and action_value != 0: # Only handle duplicates of non-zero actions
                indices = [i for i, x in enumerate(processed_actions) if x == action_value]
                # Use self.rng.choice (assuming self.rng is np.random.RandomState or similar)
                keep_map[action_value] = self.rng.choice(indices)
        
        final_actions = [
            act if act not in keep_map or i == keep_map[act] else 0
            for i, act in enumerate(processed_actions)
        ]
        return final_actions

# Removed CLIPEmbedder class definition (moved to components/embedder.py)

# Scorer model classes and factories moved to experiments/llm/models/scorer_models.py
# Import them if they were used internally in this file (they are not directly, only LLMGrid uses them via experiment.py)

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
    # Use .pt extension for torch tensors
    cache_path = os.path.join(cache_dir, f"emissions_{cache_id}.pt") 

    target_device = embedder.device # Get target device from embedder

    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading emissions from cache: {cache_path}")
        try:
            # Use torch.load with map_location
            emissions = torch.load(cache_path, map_location=target_device)
            if verbose:
                print(f"Loaded emissions tensor with shape {emissions.shape} to device {emissions.device}")
            # Basic check if loaded tensor seems valid (e.g., correct dtype, shape if known)
            if not isinstance(emissions, torch.Tensor):
                 raise TypeError("Cached file did not contain a torch.Tensor")
            # Ensure correct dtype after loading (optional but good practice)
            return emissions.double() 
        except Exception as e: # Catch broader exceptions during load/validation
            if verbose:
                print(f"Cache file corrupted or invalid ({e}), regenerating")
            # Attempt to remove corrupted cache file
            try:
                os.remove(cache_path)
            except OSError as remove_err:
                if verbose:
                    print(f"Warning: Could not remove corrupted cache file {cache_path}: {remove_err}")

    if verbose: # Print generating message only if not loaded from cache
        print("PREPROCESS: Generating emissions...")
    emissions = []
    total_elements = len(unique_elements)
    print_interval = 10  # Print progress every 10 iterations

    for i, text in enumerate(unique_elements):
        embed_text = text  # Always embed the text as is
        if verbose:
            # Verbose printing can be helpful for debugging token content
            print(f"Embedding text: {text}")

        # Use the embedder's method
        feat = embedder.embed_text(text)
        emissions.append(feat)

        # Print progress
        if verbose and ((i + 1) % print_interval == 0 or (i + 1) == total_elements):
            print(f"Generated emission {i + 1}/{total_elements}")

    if verbose: # Print done message only if generating
        print("Done generating emissions.")
    emissions = torch.vstack(emissions)

    # Ensure the generated tensor is on the correct device before saving
    emissions = emissions.to(target_device) 

    try:
        # Use torch.save to preserve tensor properties including device (implicitly)
        torch.save(emissions, cache_path)
        if verbose:
            print(f"Saved emissions cache to: {cache_path}")
    except Exception as e:
        if verbose:
            print(f"Failed to cache emissions: {e}")

    # Ensure correct dtype before returning
    return emissions.double()


# Removed setup_clip_model function (handled by embedder factory)


def create_prompt_from_tokens(tokens: List[str], base_prompt: str = '') -> str:
    """Create a prompt from a list of tokens

    Args:
        tokens: List of text tokens
        base_prompt: Optional base prompt to prepend

    Returns:
        Formatted prompt string using "in the style of" separator.
    """
    # Filter out empty tokens and strip whitespace
    valid_tokens = [str(token).strip() for token in tokens if str(token).strip()]

    if not valid_tokens:
        # If there are no valid tokens, return the base prompt or empty string
        return base_prompt

    # Simple format: base, token1, token2, ... or token1, token2, ...
    if base_prompt:
        return f"{base_prompt}, {', '.join(valid_tokens)}"
    else:
        return ', '.join(valid_tokens)


def create_prompt(actions: List[int], env: LLMGrid) -> str:
    """Create prompt string from a sequence of action indices.

    Args:
        actions: List of action indices.
        env: The LLMGrid environment instance.

    Returns:
        Formatted prompt string.
    """
    # Handle duplicate actions using the method from the env instance
    actions = env._handle_duplicate_actions(actions)

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

# get_scorer_model and make_theta_star moved to experiments/llm/models/scorer_models.py
