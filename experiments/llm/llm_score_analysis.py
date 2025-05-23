import torch
import os
import numpy as np
from experiments.llm.components.embedder import CLIPEmbedder
from doexpy.env.llm import LLMGrid, create_prompt_from_tokens # Removed DotProductModel
from experiments.llm.models.scorer_models import get_scorer_model # Import from new location
import random

# --- 1. Initialize Embedder ---
clip_embedder = CLIPEmbedder(
    model_id="openai/clip-vit-large-patch14",
    cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    normalize=True
)
print(f"CLIP Embedder initialized with model: {clip_embedder.model_id}")

# --- 2. Simplified Analysis: Fixed Base Prompt (H=1) + Appended Token (H=2) ---
print("\n--- Simplified Analysis: Fixed Base Prompt (H=1) + Appended Token (H=2) ---")

# Setup for this test
ANALYSIS_SEED = 42 # Seed for reproducibility
random.seed(ANALYSIS_SEED)
np_rng_for_analysis = np.random.RandomState(ANALYSIS_SEED)

# Define the fixed H=1 base prompt element
fixed_base_prompt_elements = [
    "A realistic photo"
]
# The LLMGrid environment's base_prompt is '' for this part.
# create_prompt_from_tokens will handle the joining.
fixed_base_prompt_str = create_prompt_from_tokens(fixed_base_prompt_elements, base_prompt='')
print(f"Using fixed H=1 base prompt: \"{fixed_base_prompt_str}\"")

# Define paths for token pools
lighting_vocab_path = './vocabulary3/lighting.txt'
ambient_vocab_path = './vocabulary3/ambient.txt'
style_vocab_path = './vocabulary3/style.txt'

token_source_paths = [lighting_vocab_path, ambient_vocab_path, style_vocab_path]
selected_appended_tokens = []
num_tokens_to_select_per_file = 100

def load_and_sample_tokens(file_path, num_to_sample):
    try:
        with open(file_path, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        if not tokens:
            print(f"Warning: {file_path} is empty or contains only whitespace. No tokens selected from this file.")
            return []
        
        if len(tokens) < num_to_sample:
            print(f"Warning: {file_path} has {len(tokens)} tokens, fewer than requested {num_to_sample}. Using all available tokens from this file.")
            return tokens
        else:
            return random.sample(tokens, num_to_sample)
            
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists and the path is correct. Exiting.")
        exit()

print("\nLoading and sampling tokens for the 5th prompt element:")
for path in token_source_paths:
    print(f"  Sampling {num_tokens_to_select_per_file} tokens from: {path}")
    sampled_tokens_from_file = load_and_sample_tokens(path, num_tokens_to_select_per_file)
    selected_appended_tokens.extend(sampled_tokens_from_file)
    print(f"    Selected {len(sampled_tokens_from_file)} tokens from this file.")

# Remove duplicates that might arise if the same token is sampled from different files (unlikely but possible if files overlap)
# Or if a file has fewer than 100 tokens and is included multiple times (not the case here).
# For this specific use case (sampling 100 from each distinct file), duplicates in the final list
# would only occur if the same token exists in multiple files and gets sampled from each.
# Keeping them distinct for now, as the request is "100 from each file".
# If truly unique tokens are needed in the final list, uncomment the next line:
# selected_appended_tokens = sorted(list(set(selected_appended_tokens)))

if not selected_appended_tokens:
    print("Error: No tokens selected from any source file. Exiting.")
    exit()

print(f"\nTotal {len(selected_appended_tokens)} tokens selected for the 5th prompt element (approx. {num_tokens_to_select_per_file} from each of {len(token_source_paths)} files):")
# Print a sample of the combined selected tokens
for token in selected_appended_tokens[:min(10, len(selected_appended_tokens))]:
    print(f"  - \"{token}\"")
if len(selected_appended_tokens) > 10:
    print(f"  ... and {len(selected_appended_tokens) - 10} more.")


# Prepare vocabulary for the LLMGrid environment (H=2)
# This is needed for the GT "sunny-image" model, which expects an environment.
# The environment structure should correspond to H=2 prompts (1 base element + 1 appended token).
# For the GT 'sunny-image' model, the vocabulary for the first step can be minimal,
# as the fixed_base_prompt_elements are used directly. The 2nd step needs the selected_appended_tokens.

# The LLMGrid for the GT model needs a vocabulary structure.
# For the first level, we provide the fixed token itself as the only option.
# For the 2nd level (appended token), we provide the selected tokens from the combined pool.
list_of_text_tokens_for_gt_env = [
    [fixed_base_prompt_elements[0]],
    selected_appended_tokens # Use the selected tokens from the combined pool for the 2nd token
]

print("\nDEBUG: About to initialize LLMGrid for GT model (H=2)...")
llm_env_for_gt = LLMGrid(
    list_of_text_tokens=list_of_text_tokens_for_gt_env,
    embedder=clip_embedder,
    base_prompt='', # Consistent with how create_prompt_from_tokens is used for full prompts
    include_base_prompt_in_first_tokens=False, # Tokens are full phrases
    rng=np_rng_for_analysis,
    verbose=False
)
# The max_episode_length of llm_env_for_gt should be 2 (len(list_of_text_tokens_for_gt_env))
print(f"LLMGrid for GT 'sunny-image' model initialized. Horizon: {llm_env_for_gt.max_episode_length}")

print("DEBUG: About to load GT 'sunny-image' model...")
# Load GT "sunny-image" model using this specific H=2 environment
gt_sunny_model = get_scorer_model(
    model_name='sunny-image',
    env=llm_env_for_gt, # Env with H=2 and selected_appended_tokens at the 2nd step
    embedder=clip_embedder
)
gt_sunny_model.eval()
print("GT 'sunny-image' model loaded.")

# Score the H=2 prompts (fixed base + selected appended token)
print(f"\nScoring {len(selected_appended_tokens)} H=2 prompts...")
results = []
for i, appended_token in enumerate(selected_appended_tokens):
    print(f"  Processing prompt {i+1}/{len(selected_appended_tokens)}: appending \"{appended_token}\"...")
    # Construct the H=2 prompt string
    full_prompt_str = f"{fixed_base_prompt_str}, {appended_token}"

    # Score with GT sunny-image model
    # The GT model's score_prompt method will internally use its env (llm_env_for_gt)
    # to potentially map tokens to indices if its internal logic requires it,
    # or use the direct prompt string for embedding if that's how it's implemented.
    # For DotProductModel, it directly embeds the prompt_text.
    gt_score_tensor, _ = gt_sunny_model.score_prompt(full_prompt_str)
    gt_score = gt_score_tensor.item()

    results.append({
        "appended_token": appended_token,
        "full_prompt": full_prompt_str,
        "gt_score": gt_score
    })

# Print results sorted by GT 'sunny-image' model score
print("\n--- Prompts Ranked by GT 'sunny-image' Model Score (Descending) ---")
results_sorted_gt = sorted(results, key=lambda x: x['gt_score'], reverse=True)
for i, res in enumerate(results_sorted_gt[:15]): # Display top 15
    print(f"  {i+1}. GT: {res['gt_score']:.4f} | Appended: \"{res['appended_token']}\"")
    # print(f"     Full Prompt: \"{res['full_prompt']}\"") # Uncomment to see full prompt
