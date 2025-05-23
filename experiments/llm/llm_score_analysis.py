import torch
import os
import numpy as np
from experiments.llm.components.embedder import CLIPEmbedder
from doexpy.env.llm import LLMGrid # Removed create_prompt_from_tokens
from experiments.llm.models.scorer_models import get_scorer_model
import random # Re-import random for adding commas

# --- Helper function to add random commas ---
def add_random_commas(sentence: str, max_commas: int = 5) -> str:
    words = sentence.split()
    if not words:
        return sentence

    # Determine the maximum number of commas to consider adding based on sentence length.
    # max_commas is the parameter from the function signature (default 5).
    
    if len(words) <= 2: # Sentences with 1 or 2 words
        max_heuristic_commas = 0
    elif len(words) <= 5: # Sentences with 3, 4, or 5 words
        max_heuristic_commas = 1
    elif len(words) <= 9: # Sentences with 6, 7, 8, or 9 words
        max_heuristic_commas = 1 # Reduced from 2
    else: # Sentences with 10 or more words
        # Cap at 2 for general naturalness, but allow max_commas to be lower if specified.
        max_heuristic_commas = 2 # Reduced from 3

    # The number of commas cannot exceed:
    # 1. The number of possible insertion points (len(words) - 1).
    # 2. The max_commas parameter passed to the function.
    # 3. Our heuristic cap based on sentence length (max_heuristic_commas).
    
    max_possible_insert_points = len(words) - 1 if len(words) > 1 else 0
    
    upper_bound_for_randint = min(max_possible_insert_points,
                                  max_commas, # from function signature
                                  max_heuristic_commas)
    
    # Ensure upper_bound_for_randint is not negative.
    upper_bound_for_randint = max(0, upper_bound_for_randint)

    num_commas_to_add = random.randint(0, upper_bound_for_randint)

    if num_commas_to_add == 0:
        return sentence

    # Possible positions for commas are between words (len(words) - 1 positions)
    # We avoid putting a comma right before the last word if it ends with a period.
    potential_indices = list(range(len(words) - 1))
    
    # If the sentence is very short, adjust potential indices
    if not potential_indices: # e.g. single word sentence
        return sentence

    comma_indices = sorted(random.sample(potential_indices, num_commas_to_add))

    new_words = []
    last_word_idx = 0
    for comma_idx in comma_indices:
        new_words.extend(words[last_word_idx:comma_idx+1])
        # Ensure no double commas or comma before period if last word
        if not new_words[-1].endswith(','):
            new_words[-1] += ','
        last_word_idx = comma_idx + 1
    new_words.extend(words[last_word_idx:])
    
    # Reconstruct sentence, ensuring spaces are correct around commas
    result_sentence = new_words[0]
    for i in range(1, len(new_words)):
        if new_words[i-1].endswith(','):
            result_sentence += " " + new_words[i]
        else: # Should not happen if commas are added correctly
            result_sentence += " " + new_words[i]
            
    # Final check to remove space before comma if any, and ensure space after
    result_sentence = result_sentence.replace(" ,", ",").replace(",", ", ")
    # Remove double spaces that might have been introduced
    result_sentence = ' '.join(result_sentence.split())
    # Ensure sentence ends with a period if original did, and no comma before period.
    if sentence.endswith('.') and result_sentence.endswith(','):
        result_sentence = result_sentence[:-1] + '.'
    elif sentence.endswith('.') and not result_sentence.endswith('.'):
        # Remove trailing comma if present before adding period
        if result_sentence.endswith(','):
            result_sentence = result_sentence[:-1].strip() + '.'
        else:
            result_sentence = result_sentence.strip() + '.'


    return result_sentence


# --- 1. Initialize Embedder ---
clip_embedder = CLIPEmbedder(
    model_id="openai/clip-vit-large-patch14",
    cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    normalize=True
)
print(f"CLIP Embedder initialized with model: {clip_embedder.model_id}")

# --- Set seed for reproducibility of comma addition ---
ANALYSIS_SEED = 42
random.seed(ANALYSIS_SEED)
# np_rng_for_analysis = np.random.RandomState(ANALYSIS_SEED) # Keep if used elsewhere, not for random.sample

# --- 2. Define Sentences ---
raw_sunny_sentences = [
    "Bright sunshine.", "Golden sunrise.", "A sunny morning.", "Warm sunshine fills the air.",
    "The early sunrise paints the sky.", "Enjoying the sunny weather.", "Sunshine streams through the window.",
    "A beautiful sunrise over the mountains.", "The cat basks in the sunny spot.", "Morning sunshine brings warmth.",
    "Watching the glorious sunrise.", "A perfect sunny afternoon.", "Golden rays of sunshine.",
    "The sun rises, a new day begins.", "Feeling the sunny glow.", "Sunshine on a clear day.",
    "The vibrant colors of sunrise.", "A day filled with bright sunshine.", "The promise of a sunny tomorrow.",
    "Early morning sunrise.", "Soaking up the warm sunshine.", "A spectacular sunrise.",
    "The park is sunny today.", "Sunshine makes everything brighter.", "The first light of sunrise.",
    "A lovely sunny day for a walk.", "The gentle touch of sunshine.", "A breathtaking sunrise view.",
    "The world awakens with the sunrise.", "Bright sunshine after the rain.", "The power of the morning sunshine.",
    "A serene and sunny landscape.", "The sky at sunrise is a masterpiece.", "Let the sunshine in.",
    "A peaceful sunny moment.", "The beauty of a daily sunrise.", "The flowers love the sunshine.",
    "A crisp sunny autumn day.", "The horizon glows at sunrise.", "Endless sunshine.",
    "The beach glowed under the afternoon sunshine.",
    "A perfect day for a picnic, with clear skies and bright sunshine.",
    "Sunlight streamed into the room, a warm and welcome sight.",
    "The valley was vibrant, kissed by the morning sunshine.",
    "Her happiness seemed to radiate like sunshine.",
    "Exploring the trails, enjoying the endless sunshine.",
    "The town square buzzed with activity under the brilliant sunshine.",
    "Sun-loving plants thrived in the garden's brightest spot.",
    "He felt refreshed by the crisp air and early sunshine.",
    "Coastal views were spectacular, enhanced by the clear sunshine."
]
sunny_sentences = [add_random_commas(s) for s in raw_sunny_sentences]

raw_non_sunny_sentences = [
    "A dark and stormy night.", "The old house stood on a lonely hill.", "Rain lashed against the windows.",
    "A quiet library filled with books.", "The deep ocean hides many secrets.", "A bustling city street at midnight.",
    "Snow covered the silent forest.", "The mystery of the ancient ruins.", "A cozy fireplace on a cold evening.",
    "The vastness of outer space.", "A thrilling adventure in the mountains.", "The sound of a distant train.",
    "A cup of hot tea on a rainy day.", "Exploring a hidden cave.", "The eerie silence of a foggy morning.",
    "A complex mathematical equation.", "The taste of freshly baked bread.", "A powerful symphony orchestra.",
    "The intricate design of a spider's web.", "A challenging game of chess.", "The scent of pine trees in the woods.",
    "A historical novel about ancient Rome.", "The feeling of soft sand between your toes.",
    "A debate about philosophical concepts.", "The structure of a protein molecule.", "A quiet evening spent reading.",
    "The mechanics of a clock.", "A recipe for a delicious cake.", "The rules of a complex board game.",
    "A documentary about wildlife.", "The art of pottery making.", "Learning a new programming language.",
    "The history of the internet.", "A discussion on economic theories.",
    "The old cat slept by the quiet window.",
    "A painting of distant hills hung in the room.",
    "The park was empty after the evening rain.",
    "She found flowers pressed in an old book.",
    "The beach was cool and misty at dawn.",
    "A quiet meadow under a grey sky.",
    "The valley was filled with a thick morning fog.",
    "He remembered the laughter from the distant adventures.",
    "The land stretched out, vast and silent under the clouds.",
    "A comfortable seat by the unlit fireplace.",
    "The water in the lake was dark and still.",
    "A single star was visible through a break in the clouds.",
    "The leaves rustled in the cool night breeze.",
    "His disposition was calm, even during the storm.",
    "The desert at night held a profound silence."
]
non_sunny_sentences = [add_random_commas(s) for s in raw_non_sunny_sentences]

all_sentences = sunny_sentences + non_sunny_sentences
print(f"Defined {len(sunny_sentences)} sunny sentences (with random commas) and {len(non_sunny_sentences)} non-sunny sentences (with random commas).")

# --- 3. Setup Environment and Scorer Model ---
# For the 'sunny' model, the LLMGrid's vocabulary structure is not critical
# as get_scorer_model directly uses the sentences from 'sunny.txt'.
# We initialize it with a minimal vocabulary.
minimal_vocab_for_env = [["placeholder"]]
llm_env_for_gt = LLMGrid(
    list_of_text_tokens=minimal_vocab_for_env,
    embedder=clip_embedder,
    base_prompt='',
    include_base_prompt_in_first_tokens=False,
    rng=np.random.RandomState(42), # Provide an RNG for LLMGrid
    verbose=False
)
print(f"Minimal LLMGrid for GT 'sunny' model initialized. Horizon: {llm_env_for_gt.max_episode_length}")

print("Loading GT 'sunny' model...")
gt_sunny_model = get_scorer_model(
    model_name='sunny',
    env=llm_env_for_gt, # Env is used by get_scorer_model to find 'sunny.txt'
    embedder=clip_embedder
)
gt_sunny_model.eval()
print("GT 'sunny' model loaded.")

# --- 4. Score all sentences ---
print(f"\nScoring {len(all_sentences)} sentences with the 'sunny' model...")
sentence_scores = {}
for i, sentence in enumerate(all_sentences):
    if (i + 1) % 10 == 0 or i == len(all_sentences) - 1:
        print(f"  Scoring sentence {i+1}/{len(all_sentences)}: \"{sentence[:50]}...\"")
    score_tensor, _ = gt_sunny_model.score_prompt(sentence)
    sentence_scores[sentence] = score_tensor.item()

# --- 5. Ranking Analysis ---
print("\n--- Ranking Analysis ---")
correct_rankings = 0
total_pairs = 0

for sunny_sent in sunny_sentences:
    for non_sunny_sent in non_sunny_sentences:
        total_pairs += 1
        score_sunny = sentence_scores[sunny_sent]
        score_non_sunny = sentence_scores[non_sunny_sent]

        if score_sunny > score_non_sunny:
            correct_rankings += 1

accuracy = (correct_rankings / total_pairs) if total_pairs > 0 else 0.0
print(f"Number of correctly ranked pairs (sunny_score > non_sunny_score): {correct_rankings}")
print(f"Total pairs compared: {total_pairs}")
print(f"Ranking Accuracy: {accuracy:.4f}")

# Optional: Print top/bottom scored sentences from each category for qualitative check
print("\n--- Top 5 Scored Sunny Sentences ---")
sorted_sunny_scores = sorted([(s, sentence_scores[s]) for s in sunny_sentences], key=lambda x: x[1], reverse=True)
for i, (sent, score) in enumerate(sorted_sunny_scores[:5]):
    print(f"  {i+1}. Score: {score:.4f} | Sentence: \"{sent}\"")

print("\n--- Top 5 Scored Non-Sunny Sentences (should be low scores) ---")
sorted_non_sunny_scores = sorted([(s, sentence_scores[s]) for s in non_sunny_sentences], key=lambda x: x[1], reverse=True)
for i, (sent, score) in enumerate(sorted_non_sunny_scores[:5]):
    print(f"  {i+1}. Score: {score:.4f} | Sentence: \"{sent}\"")

print("\n--- Bottom 5 Scored Non-Sunny Sentences (should be very low scores) ---")
for i, (sent, score) in enumerate(sorted_non_sunny_scores[-5:]):
    print(f"  {len(sorted_non_sunny_scores)-5+i+1}. Score: {score:.4f} | Sentence: \"{sent}\"")
