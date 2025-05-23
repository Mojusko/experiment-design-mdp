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

# --- 3. Load sentences from sunny.txt and embed all relevant sentences ---
sunny_txt_path = os.path.join(os.path.dirname(__file__), 'models', 'sunny.txt') # Corrected path
try:
    with open(sunny_txt_path, 'r') as f:
        raw_sentences_from_sunny_txt = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(raw_sentences_from_sunny_txt)} sentences from {sunny_txt_path}")
except FileNotFoundError:
    print(f"Error: sunny.txt not found at {sunny_txt_path}")
    raw_sentences_from_sunny_txt = []

print(f"\nEmbedding {len(raw_sentences_from_sunny_txt)} sentences from sunny.txt...")
embeddings_from_sunny_txt = []
for i, sentence in enumerate(raw_sentences_from_sunny_txt):
    if (i + 1) % 5 == 0 or i == len(raw_sentences_from_sunny_txt) - 1:
        print(f"  Embedding sentence {i+1}/{len(raw_sentences_from_sunny_txt)} from sunny.txt: \"{sentence[:50]}...\"")
    embeddings_from_sunny_txt.append(clip_embedder.embed_text(sentence))

# Embed script's sunny sentences (processed with commas)
print(f"\nEmbedding {len(sunny_sentences)} processed sunny sentences (from script)...")
script_sunny_embeddings = []
for i, sentence in enumerate(sunny_sentences): # sunny_sentences is already processed with commas
    if (i + 1) % 10 == 0 or i == len(sunny_sentences) - 1:
        print(f"  Embedding script's sunny sentence {i+1}/{len(sunny_sentences)}: \"{sentence[:50]}...\"")
    script_sunny_embeddings.append(clip_embedder.embed_text(sentence))

print(f"\nEmbedding {len(non_sunny_sentences)} processed non-sunny sentences...")
embeddings_non_sunny = []
for i, sentence in enumerate(non_sunny_sentences): # non_sunny_sentences is already processed with commas
    if (i + 1) % 10 == 0 or i == len(non_sunny_sentences) - 1:
        print(f"  Embedding non-sunny sentence {i+1}/{len(non_sunny_sentences)}: \"{sentence[:50]}...\"")
    embeddings_non_sunny.append(clip_embedder.embed_text(sentence))

# --- 4. Overall Ranking Analysis ---
print("\n--- Overall Ranking Analysis (Each sunny.txt sentence as a model) ---")
correct_rankings_overall = 0
total_pairs_overall = 0

# Check if all necessary embeddings lists are populated
if embeddings_from_sunny_txt and script_sunny_embeddings and embeddings_non_sunny:
    # Iterate through each sentence from sunny.txt (as a model)
    for i, model_theta_embedding in enumerate(embeddings_from_sunny_txt):
        # Iterate through each sunny sentence from the script (as a positive example)
        for j, script_sunny_emb in enumerate(script_sunny_embeddings):
            score_positive_by_model = torch.dot(model_theta_embedding.squeeze(), script_sunny_emb.squeeze()).item()
            
            # Iterate through each non-sunny sentence from the script (as a negative example)
            for k, script_non_sunny_emb in enumerate(embeddings_non_sunny):
                score_negative_by_model = torch.dot(model_theta_embedding.squeeze(), script_non_sunny_emb.squeeze()).item()
                
                if score_positive_by_model > score_negative_by_model:
                    correct_rankings_overall += 1
                total_pairs_overall += 1
        
        # Print progress for each model processed from sunny.txt
        if (i + 1) % 1 == 0 or i == len(embeddings_from_sunny_txt) - 1: # Adjusted print frequency
            print(f"  Processed model {i+1}/{len(embeddings_from_sunny_txt)} from sunny.txt for overall accuracy. Current total pairs: {total_pairs_overall}")

accuracy_overall = (correct_rankings_overall / total_pairs_overall) if total_pairs_overall > 0 else 0.0
print(f"Number of correctly ranked pairs (overall): {correct_rankings_overall}")
print(f"Total pairs compared (overall): {total_pairs_overall}")
print(f"Overall Ranking Accuracy: {accuracy_overall:.4f}")


# --- 5. Individual Sentence Accuracy Analysis (using sentences from sunny.txt) ---
print("\n--- Individual Sunny Sentence Accuracies (from sunny.txt) ---")

def calculate_single_concept_accuracy(
    emb_sunny_theta_model,  # Embedding of the target sunny sentence from sunny.txt (acts as theta for this model)
    local_script_sunny_embeddings_list, # List of embeddings of processed sunny sentences from the script
    local_script_non_sunny_embeddings_list # List of embeddings of processed non-sunny sentences from the script
):
    correct_single = 0
    num_comparison_pairs = 0
    # emb_sunny_theta_model is the vector defining the current model.
    
    for emb_script_sunny in local_script_sunny_embeddings_list:
        score_positive_example_by_model = torch.dot(emb_sunny_theta_model.squeeze(), emb_script_sunny.squeeze()).item()
        
        for emb_script_non_sunny in local_script_non_sunny_embeddings_list:
            score_negative_example_by_model = torch.dot(emb_sunny_theta_model.squeeze(), emb_script_non_sunny.squeeze()).item()
            
            if score_positive_example_by_model > score_negative_example_by_model:
                correct_single += 1
            num_comparison_pairs += 1
            
    return correct_single / num_comparison_pairs if num_comparison_pairs > 0 else 0.0


if raw_sentences_from_sunny_txt and embeddings_from_sunny_txt and script_sunny_embeddings and embeddings_non_sunny:
    # 1. Accuracy for the first sentence from sunny.txt
    target_raw_sentence_first = raw_sentences_from_sunny_txt[0]
    target_emb_first = embeddings_from_sunny_txt[0] # This is the theta for the first model
    acc_first = calculate_single_concept_accuracy(target_emb_first, script_sunny_embeddings, embeddings_non_sunny)
    print(f"'{target_raw_sentence_first}' - accuracy: {acc_first:.4f}")

    # 2. Accuracies for the next 19 sentences from sunny.txt (indices 1 to 19)
    for i in range(1, 20): # Iterate for sentences at index 1 through 19 in sunny.txt list
        if i < len(raw_sentences_from_sunny_txt):
            current_raw_sentence_from_txt = raw_sentences_from_sunny_txt[i]
            current_emb_from_sunny_txt_as_model = embeddings_from_sunny_txt[i] # This is theta for the current model
            acc_current = calculate_single_concept_accuracy(current_emb_from_sunny_txt_as_model, script_sunny_embeddings, embeddings_non_sunny)
            print(f"'{current_raw_sentence_from_txt}' - accuracy: {acc_current:.4f}")
        else:
            # This means sunny.txt has fewer than (i+1) sentences.
            print(f"Warning: Index {i} out of bounds for raw_sentences_from_sunny_txt (length {len(raw_sentences_from_sunny_txt)}).")
            break # Stop if we run out of sentences
else:
    print("Skipping individual sentence accuracy calculation due to missing sentences or embeddings.")
