import torch
import os
import numpy as np
from experiments.llm.components.embedder import CLIPEmbedder
from doexpy.env.llm import DotProductModel, LLMGrid, get_scorer_model, create_prompt_from_tokens
import random

# --- 1. Load Estimator Theta ---
estimator_file_path = "./results/feedback-comparison-2025-05-22-13-51/estimator-sunny-feedback-dsn-mult-ep160-1.pt"

loaded_estimator_theta = torch.load(estimator_file_path, map_location=torch.device('cpu'))
print(f"Loaded estimator theta from: {estimator_file_path}, shape: {loaded_estimator_theta.shape}")

# --- 2. Initialize Embedder ---
clip_embedder = CLIPEmbedder(
    model_id="openai/clip-vit-large-patch14",
    cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    normalize=True
)
print(f"CLIP Embedder initialized with model: {clip_embedder.model_id}")

# --- 3. Create Scoring Model (from loaded estimator) ---
estimator_scoring_model = DotProductModel(
    embedder=clip_embedder,
    weight=loaded_estimator_theta
)
estimator_scoring_model.eval()
print("Estimator-based scoring model created.")

# --- 4. Simplified Analysis: Fixed Base Prompt + 10 Random Tokens from lighting.txt ---
print("\n--- Simplified Analysis: Fixed Base Prompt + 10 Random Tokens from lighting.txt ---")

# Setup for this test
ANALYSIS_SEED = 42 # Seed for reproducibility
random.seed(ANALYSIS_SEED)
np_rng_for_analysis = np.random.RandomState(ANALYSIS_SEED)

# Define the fixed H=4 base prompt elements
fixed_base_prompt_elements = [
    "A realistic photo",
    "a serene landscape",
    "in the style of Ansel Adams",
    "with a wide angle lens"
]
# The LLMGrid environment's base_prompt is '' for this part.
# create_prompt_from_tokens will handle the joining.
fixed_base_prompt_str = create_prompt_from_tokens(fixed_base_prompt_elements, base_prompt='')
print(f"Using fixed H=4 base prompt: \"{fixed_base_prompt_str}\"")

# Load tokens from lighting.txt for appending
lighting_vocab_path = './vocabulary3/lighting.txt'
try:
    with open(lighting_vocab_path, 'r') as f:
        lighting_tokens_pool = [line.strip() for line in f if line.strip()]
    if not lighting_tokens_pool:
        raise ValueError("lighting.txt is empty or contains only whitespace.")
except FileNotFoundError:
    print(f"Error: lighting.txt not found at {lighting_vocab_path}. Please ensure the file exists and the path is correct. Exiting.")
    exit()
except ValueError as e:
    print(f"Error: {e}. Exiting.")
    exit()

# Select 10 random tokens from lighting.txt
num_tokens_to_select = 100
if len(lighting_tokens_pool) < num_tokens_to_select:
    print(f"Warning: lighting.txt has {len(lighting_tokens_pool)} tokens, which is fewer than the requested {num_tokens_to_select}. Using all available tokens.")
    selected_lighting_tokens = lighting_tokens_pool
else:
    selected_lighting_tokens = random.sample(lighting_tokens_pool, num_tokens_to_select)

print(f"\nSelected {len(selected_lighting_tokens)} random tokens from '{lighting_vocab_path}':")
for token in selected_lighting_tokens:
    print(f"  - \"{token}\"")

# Prepare vocabulary for the LLMGrid environment (H=5)
# This is needed for the GT "sunny" model, which expects an environment.
# The environment structure should correspond to H=5 prompts (4 base elements + 1 lighting token).
# For the GT 'sunny' model, the vocabulary for the first 4 steps can be minimal,
# as the fixed_base_prompt_elements are used directly. The 5th step needs the lighting_tokens_pool.

# Placeholder paths for vocab files corresponding to the fixed base prompt elements.
# These are only used to structure the list_of_text_tokens for LLMGrid.
# The actual content for these steps comes from fixed_base_prompt_elements.
placeholder_base_vocab_path = './llm/vocabulary3/bases.txt' # Actual content not critical if fixed_base_prompt_elements[0] is used
placeholder_ambient_vocab_path = './llm/vocabulary3/ambient.txt'
placeholder_style_vocab_path = './llm/vocabulary3/style.txt'
placeholder_composition_vocab_path = './llm/vocabulary3/composition.txt'

VOCAB_FILES_ORDERED_KEYS_FOR_GT_ENV = [
    placeholder_base_vocab_path,
    placeholder_ambient_vocab_path,
    placeholder_style_vocab_path,
    placeholder_composition_vocab_path,
    lighting_vocab_path  # Crucial: This must be the actual path to lighting.txt
]

# The LLMGrid for the GT model needs a vocabulary structure.
# For the first 4 levels, we can provide the fixed token itself as the only option.
# For the 5th level (lighting), we provide the full pool from lighting.txt.
list_of_text_tokens_for_gt_env = [
    [fixed_base_prompt_elements[0]],
    [fixed_base_prompt_elements[1]],
    [fixed_base_prompt_elements[2]],
    [fixed_base_prompt_elements[3]],
    selected_lighting_tokens # Use only the 10 selected tokens for the 5th token
]

print("DEBUG: About to initialize LLMGrid for GT model...")
llm_env_for_gt = LLMGrid(
    list_of_text_tokens=list_of_text_tokens_for_gt_env,
    embedder=clip_embedder,
    base_prompt='', # Consistent with how create_prompt_from_tokens is used for full prompts
    include_base_prompt_in_first_tokens=False, # Tokens are full phrases
    rng=np_rng_for_analysis,
    verbose=False
)
# The max_episode_length of llm_env_for_gt should be 5 (len(list_of_text_tokens_for_gt_env))
print(f"\nLLMGrid for GT 'sunny' model initialized. Horizon: {llm_env_for_gt.max_episode_length}")

print("DEBUG: About to load GT 'sunny' model...")
# Load GT "sunny" model using this specific H=5 environment
gt_sunny_model = get_scorer_model(
    model_name='sunny',
    env=llm_env_for_gt, # Env with H=5 and lighting_tokens_pool at the 5th step
    embedder=clip_embedder
)
gt_sunny_model.eval()
print("GT 'sunny' model loaded.")

# Score the H=5 prompts (fixed base + selected lighting token)
print(f"\nScoring {len(selected_lighting_tokens)} H=5 prompts...")
results = []
for i, lighting_token in enumerate(selected_lighting_tokens):
    print(f"  Processing prompt {i+1}/{len(selected_lighting_tokens)}: appending \"{lighting_token}\"...")
    # Construct the H=5 prompt string
    full_prompt_str = f"{fixed_base_prompt_str}, {lighting_token}"

    # Score with estimator model
    est_score_tensor, _ = estimator_scoring_model.score_prompt(full_prompt_str)
    est_score = est_score_tensor.item()

    # Score with GT sunny model
    # The GT model's score_prompt method will internally use its env (llm_env_for_gt)
    # to potentially map tokens to indices if its internal logic requires it,
    # or use the direct prompt string for embedding if that's how it's implemented.
    # For DotProductModel, it directly embeds the prompt_text.
    gt_score_tensor, _ = gt_sunny_model.score_prompt(full_prompt_str)
    import ipdb; ipdb.set_trace()
    gt_score = gt_score_tensor.item()

    results.append({
        "appended_token": lighting_token,
        "full_prompt": full_prompt_str,
        "estimator_score": est_score,
        "gt_score": gt_score
    })

# Print results sorted by GT 'sunny' model score
print("\n--- Prompts Ranked by GT 'sunny' Model Score (Descending) ---")
results_sorted_gt = sorted(results, key=lambda x: x['gt_score'], reverse=True)
for i, res in enumerate(results_sorted_gt[:15]): # Display top 15
    print(f"  {i+1}. GT: {res['gt_score']:.4f}, Est: {res['estimator_score']:.4f} | Appended: \"{res['appended_token']}\"")
    # print(f"     Full Prompt: \"{res['full_prompt']}\"") # Uncomment to see full prompt

# Print results sorted by Estimator model score
print("\n--- Prompts Ranked by Estimator Model Score (Descending) ---")
results_sorted_est = sorted(results, key=lambda x: x['estimator_score'], reverse=True)
for i, res in enumerate(results_sorted_est[:15]): # Display top 15
    print(f"  {i+1}. Est: {res['estimator_score']:.4f}, GT: {res['gt_score']:.4f} | Appended: \"{res['appended_token']}\"")
    # print(f"     Full Prompt: \"{res['full_prompt']}\"") # Uncomment to see full prompt
