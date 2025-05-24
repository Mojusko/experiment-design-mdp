import torch
import os
import numpy as np
import argparse # Added argparse
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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Analyze sentence scoring for a given model.")
parser.add_argument(
    "--model_name",
    type=str,
    default="sunny",
    help="Name of the model to analyze (e.g., 'sunny', 'medieval'). Corresponds to a .txt file in experiments/llm/models/."
)
args = parser.parse_args()
model_name_to_analyze = args.model_name
print(f"Analyzing model: {model_name_to_analyze}")

# --- Set seed for reproducibility of comma addition ---
ANALYSIS_SEED = 42
random.seed(ANALYSIS_SEED)
# np_rng_for_analysis = np.random.RandomState(ANALYSIS_SEED) # Keep if used elsewhere, not for random.sample

# --- 2. Define Sentences ---
raw_sunny_sentences = [ # These will now be POSITIVE TECHNOLOGICAL sentences
    "An image of a gleaming android with visible circuitry, analyzing data on a transparent screen.",
    "An image with a swarm of miniature drones, forming a complex aerial pattern.",
    "An image of a futuristic city skyline, with towering skyscrapers and flying vehicles.",
    "An image with a person wearing a neural interface headset, interacting with a virtual environment.",
    "An image of a high-speed maglev train, streaking through a technologically advanced landscape.",
    "An image with robotic arms assembling intricate microchips in a sterile factory.",
    "An image of a space elevator, connecting Earth to an orbital station.",
    "An image with genetically engineered crops, glowing faintly under artificial sunlamps in a vertical farm.",
    "An image of a powered exoskeleton suit, enhancing human strength and agility.",
    "An image with a quantum computer, its complex core visible with cryogenic cooling systems.",
    "An image of a self-assembling nanobot colony, constructing a microscopic device.",
    "An image with advanced medical scanners, providing detailed internal body views.",
    "An image of a fusion reactor core, glowing with contained plasma energy.",
    "An image with augmented reality glasses, overlaying digital information onto the real world.",
    "An image of a bionic limb, seamlessly integrated with a human user.",
    "An image with laser communication arrays, transmitting data across vast interstellar distances.",
    "An image of a terraforming machine, altering the atmosphere of a barren planet.",
    "An image with smart dust particles, collecting environmental data across a wide area.",
    "An image of an anti-gravity vehicle, hovering silently above the ground.",
    "An image with a digital consciousness, represented as a flowing stream of light and data.",
    "An image of a cybernetically enhanced animal, equipped with technological implants.",
    "An image with advanced 3D food printers, creating customized meals layer by layer.",
    "An image of a personal energy shield, deflecting an incoming projectile.",
    "An image with sophisticated surveillance satellites, orbiting high above the Earth.",
    "An image of a deep-sea exploration mech, navigating an abyssal trench.",
    "An image with holographic advertisements, shimmering on city buildings.",
    "An image of a synthetic biology lab, with scientists designing new life forms.",
    "An image with wearable technology, displaying vital signs and communication interfaces.",
    "An image of an automated mining operation on an asteroid, run by AI-controlled robots.",
    "An image with advanced cloaking technology, rendering an object nearly invisible.",
    "An image of a Dyson swarm segment, partially enclosing a distant star to harvest energy.",
    "An image with sonic weaponry, emitting focused sound waves.",
    "An image of a virtual reality classroom, with students interacting as avatars.",
    "An image with AI-powered diagnostic tools, identifying diseases from medical images.",
    "An image of a climate control system, managing weather patterns over a large city.",
    "An image with advanced water purification technology, turning desert air into potable water.",
    "An image of a thought-controlled prosthetic, allowing intuitive movement.",
    "An image with self-repairing materials, mending cracks in a futuristic structure.",
    "An image of a global data network, visualized as interconnected nodes of light.",
    "An image with advanced sensor networks, monitoring a complex ecosystem.",
    "An image of a plasma rifle, glowing with contained energy before firing.",
    "An image with orbital solar power collectors, beaming energy down to Earth.",
    "An image of a bio-luminescent data interface, tattooed onto a user's arm.",
    "An image with cryosleep pods, for long-duration space travel.",
    "An image of an AI artist, creating a complex digital painting with robotic arms.",
    "An image with force field barriers, protecting a secure facility.",
    "An image of a molecular assembler, constructing objects atom by atom.",
    "An image with advanced holographic communication, showing a life-sized 3D projection.",
    "An image of a personal flying drone, used for urban transportation.",
    "An image with a brain-computer interface, allowing direct thought-to-text typing.",
]
sunny_sentences = [add_random_commas(s) for s in raw_sunny_sentences]

raw_non_sunny_sentences = [ # These will now be TRICKY NEGATIVE (NON-TECHNOLOGICAL) sentences
    "An image of an ancient water clock, meticulously designed with gears and water flow.",
    "An image with intricate patterns of frost on a window pane, resembling circuit boards.",
    "An image of a complex spider web, glistening with dew drops in the morning light.",
    "An image with a detailed anatomical illustration of the human nervous system from a 19th-century textbook.",
    "An image of a natural crystal formation, with sharp geometric angles and facets.",
    "An image with a flock of birds, flying in a perfectly synchronized V-formation.",
    "An image of a blacksmith's forge, with bellows and tools for shaping metal with fire.",
    "An image with an old, complex mechanical music box, playing a delicate tune.",
    "An image of a detailed map of a city's subway system, showing interconnected lines.",
    "An image with a beehive's internal structure, showcasing hexagonal honeycomb cells.",
    "An image of a vintage telegraph machine, with brass keys and wiring.",
    "An image with a meticulously arranged Japanese rock garden, symbolizing natural landscapes.",
    "An image of an old film projector, casting flickering images onto a screen.",
    "An image with a complex knot, tied with precision for a nautical application.",
    "An image of a sundial, accurately telling time by the shadow of the sun.",
    "An image with a weaver's loom, threaded with colorful yarns for intricate textile patterns.",
    "An image of a printing press from the Gutenberg era, with movable type.",
    "An image with a collection of antique scientific instruments, like astrolabes and sextants.",
    "An image of a detailed schematic for a steam engine, showing pistons and valves.",
    "An image with naturally occurring geometric patterns in a snowflake, viewed under a microscope.",
    "An image of an abacus, used for complex calculations with beads on rods.",
    "An image with a lighthouse's Fresnel lens, concentrating light into a powerful beam.",
    "An image of a traditional windmill, with large sails turning to grind grain.",
    "An image with an old, ornate cash register, with mechanical buttons and a bell.",
    "An image of a complex system of irrigation canals, built by an ancient civilization.",
    "An image with a detailed drawing of a bird's wing, showing its aerodynamic structure.",
    "An image of a vintage radio, with vacuum tubes and a large tuning dial.",
    "An image with a carefully constructed beaver dam, altering the flow of a river.",
    "An image of a locksmith's tools, designed for manipulating intricate lock mechanisms.",
    "An image with a player piano, using a perforated paper roll to play music automatically.",
    "An image of a complex board game, with many pieces and a detailed game board.",
    "An image with an old camera obscura, projecting an image of the outside world.",
    "An image of a traditional water wheel, powering a mill or workshop.",
    "An image with a set of tuning forks, each producing a precise musical pitch.",
    "An image of a complex origami creation, folded from a single sheet of paper.",
    "An image with an old ship's rigging, a complex network of ropes and pulleys.",
    "An image of a seismograph, recording earth tremors with a needle on a drum.",
    "An image with a vintage typewriter, its keys and mechanical arms poised to strike.",
    "An image of a detailed architectural model of a historical building, made from wood.",
    "An image with a system of gears and levers in an old grandfather clock.",
    "An image of a coral reef, an intricate ecosystem built by tiny organisms.",
    "An image with a cartographer's tools, used for drawing precise maps by hand.",
    "An image of a traditional pottery wheel, shaping clay with skilled hands.",
    "An image with an old slide rule, used for mathematical computations before calculators.",
    "An image of a complex ant colony, with tunnels and chambers visible in a cross-section.",
    "An image with a vintage sewing machine, with intricate mechanical parts.",
    "An image of a human eye, showing the complex structure of the iris and pupil.",
    "An image with a musical score, filled with complex notation for an orchestra.",
    "An image of a bird's nest, intricately woven from twigs and other natural materials.",
    "An image with an old, hand-cranked telephone, connecting to a manual switchboard.",
]
non_sunny_sentences = [add_random_commas(s) for s in raw_non_sunny_sentences]

all_sentences = sunny_sentences + non_sunny_sentences
print(f"Defined {len(sunny_sentences)} '{model_name_to_analyze}' sentences (with random commas) and {len(non_sunny_sentences)} non-'{model_name_to_analyze}' sentences (with random commas).")

# --- 3. Load sentences from {model_name_to_analyze}.txt and embed all relevant sentences ---
model_txt_filename = f"{model_name_to_analyze}.txt"
model_txt_path = os.path.join(os.path.dirname(__file__), 'models', model_txt_filename)
try:
    with open(model_txt_path, 'r') as f:
        raw_sentences_from_model_txt = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(raw_sentences_from_model_txt)} sentences from {model_txt_path}")
except FileNotFoundError:
    print(f"Error: {model_txt_filename} not found at {model_txt_path}")
    raw_sentences_from_model_txt = []

print(f"\nEmbedding {len(raw_sentences_from_model_txt)} sentences from {model_txt_filename}...")
embeddings_from_model_txt = []
for i, sentence in enumerate(raw_sentences_from_model_txt):
    if (i + 1) % 5 == 0 or i == len(raw_sentences_from_model_txt) - 1:
        print(f"  Embedding sentence {i+1}/{len(raw_sentences_from_model_txt)} from {model_txt_filename}: \"{sentence[:50]}...\"")
    embeddings_from_model_txt.append(clip_embedder.embed_text(sentence))

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
print(f"\n--- Overall Ranking Analysis (Each {model_name_to_analyze}.txt sentence as a model) ---")
correct_rankings_overall = 0
total_pairs_overall = 0

# Check if all necessary embeddings lists are populated
if embeddings_from_model_txt and script_sunny_embeddings and embeddings_non_sunny:
    # Iterate through each sentence from {model_name_to_analyze}.txt (as a model)
    for i, model_theta_embedding in enumerate(embeddings_from_model_txt):
        # Iterate through each '{model_name_to_analyze}' sentence from the script (as a positive example)
        for j, script_positive_emb in enumerate(script_sunny_embeddings): # script_sunny_embeddings now holds "medieval" etc.
            score_positive_by_model = torch.dot(model_theta_embedding.squeeze(), script_positive_emb.squeeze()).item()
            
            # Iterate through each non-'{model_name_to_analyze}' sentence from the script (as a negative example)
            for k, script_non_positive_emb in enumerate(embeddings_non_sunny): # embeddings_non_sunny now holds "non-medieval" etc.
                score_negative_by_model = torch.dot(model_theta_embedding.squeeze(), script_non_positive_emb.squeeze()).item()
                
                if score_positive_by_model > score_negative_by_model:
                    correct_rankings_overall += 1
                total_pairs_overall += 1
        
        # Print progress for each model processed from {model_name_to_analyze}.txt
        if (i + 1) % 1 == 0 or i == len(embeddings_from_model_txt) - 1: # Adjusted print frequency
            print(f"  Processed model {i+1}/{len(embeddings_from_model_txt)} from {model_name_to_analyze}.txt for overall accuracy. Current total pairs: {total_pairs_overall}")

accuracy_overall = (correct_rankings_overall / total_pairs_overall) if total_pairs_overall > 0 else 0.0
print(f"Number of correctly ranked pairs (overall): {correct_rankings_overall}")
print(f"Total pairs compared (overall): {total_pairs_overall}")
print(f"Overall Ranking Accuracy: {accuracy_overall:.4f}")


# --- 5. Individual Sentence Accuracy Analysis (using sentences from {model_name_to_analyze}.txt) ---
print(f"\n--- Individual {model_name_to_analyze.capitalize()} Sentence Accuracies (from {model_name_to_analyze}.txt) ---")

def calculate_single_concept_accuracy(
    emb_model_theta,  # Embedding of the target sentence from {model_name_to_analyze}.txt (acts as theta for this model)
    local_script_positive_embeddings_list, # List of embeddings of processed positive sentences from the script
    local_script_negative_embeddings_list # List of embeddings of processed negative sentences from the script
):
    correct_single = 0
    num_comparison_pairs = 0
    # emb_model_theta is the vector defining the current model.
    
    for emb_script_positive in local_script_positive_embeddings_list:
        score_positive_example_by_model = torch.dot(emb_model_theta.squeeze(), emb_script_positive.squeeze()).item()
        
        for emb_script_negative in local_script_negative_embeddings_list:
            score_negative_example_by_model = torch.dot(emb_model_theta.squeeze(), emb_script_negative.squeeze()).item()
            
            if score_positive_example_by_model > score_negative_example_by_model:
                correct_single += 1
            num_comparison_pairs += 1
            
    return correct_single / num_comparison_pairs if num_comparison_pairs > 0 else 0.0


if raw_sentences_from_model_txt and embeddings_from_model_txt and script_sunny_embeddings and embeddings_non_sunny:
    # 1. Accuracy for the first sentence from {model_name_to_analyze}.txt
    target_raw_sentence_first = raw_sentences_from_model_txt[0]
    target_emb_first = embeddings_from_model_txt[0] # This is the theta for the first model
    acc_first = calculate_single_concept_accuracy(target_emb_first, script_sunny_embeddings, embeddings_non_sunny)
    print(f"'{target_raw_sentence_first}' - accuracy: {acc_first:.4f}")

    # 2. Accuracies for the next 19 sentences from {model_name_to_analyze}.txt (indices 1 to 19)
    for i in range(1, 20): # Iterate for sentences at index 1 through 19 in {model_name_to_analyze}.txt list
        if i < len(raw_sentences_from_model_txt):
            current_raw_sentence_from_txt = raw_sentences_from_model_txt[i]
            current_emb_from_model_txt_as_model = embeddings_from_model_txt[i] # This is theta for the current model
            acc_current = calculate_single_concept_accuracy(current_emb_from_model_txt_as_model, script_sunny_embeddings, embeddings_non_sunny)
            print(f"'{current_raw_sentence_from_txt}' - accuracy: {acc_current:.4f}")
        else:
            # This means {model_name_to_analyze}.txt has fewer than (i+1) sentences.
            print(f"Warning: Index {i} out of bounds for raw_sentences_from_model_txt (length {len(raw_sentences_from_model_txt)}).")
            break # Stop if we run out of sentences
else:
    print("Skipping individual sentence accuracy calculation due to missing sentences or embeddings.")
