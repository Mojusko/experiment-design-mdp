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
raw_positive_medieval_sentences = [ # These are now POSITIVE MEDIEVAL sentences
    "An image of a knight in shining armor, wielding a sword and shield before a castle.",
    "An image of a grand medieval castle, with tall stone towers, battlements, and a wide moat.",
    "An image of a bustling medieval marketplace, filled with merchants, craftsmen, and colorful stalls.",
    "An image of a king and queen on ornate thrones, in a grand medieval hall adorned with tapestries.",
    "An image of a jousting tournament, with two knights on horseback charging with lances.",
    "An image of a medieval village, with timber-framed houses, thatched roofs, and cobblestone streets.",
    "An image of a scribe in a monastery scriptorium, diligently illuminating a manuscript with gold leaf.",
    "An image of a medieval catapult, launching a large stone towards castle walls during a siege.",
    "An image of a fearsome dragon, perched atop a craggy mountain, overlooking a medieval kingdom.",
    "An image of a medieval feast, with long wooden tables laden with roasted meats, bread, and goblets.",
    "An image of archers on castle ramparts, drawing their longbows, aiming at an approaching enemy.",
    "An image of a medieval blacksmith at his forge, hammering glowing metal on an anvil.",
    "An image of a magnificent stained glass window, in a Gothic cathedral, depicting a biblical scene.",
    "An image of a medieval tapestry, intricately woven, showing a heroic battle or a courtly scene.",
    "An image of a wise wizard in a tall tower, surrounded by ancient books, bubbling potions, and mystical artifacts.",
    "An image of a medieval sailing ship, with large square sails, navigating a stormy, dark sea.",
    "An image of peasants toiling in the fields, harvesting crops with simple tools, a distant castle on the horizon.",
    "An image of a powerful medieval trebuchet, its arm swinging to hurl a massive projectile.",
    "An image of a royal procession, with nobles in fine attire, knights on horseback, and colorful banners.",
    "An image of a tall wooden siege tower, slowly advancing towards the fortified walls of a city.",
    "An image of a medieval minstrel, playing a lute and singing ballads in a castle courtyard.",
    "An image of a fortified medieval bridge, with guard towers and a portcullis, spanning a river.",
    "An image of a medieval alchemist, in a cluttered laboratory, attempting to transmute lead into gold.",
    "An image of a group of pilgrims, journeying on foot along a dusty road to a holy shrine.",
    "An image of a medieval armory, filled with suits of armor, swords, shields, and other weapons.",
    "An image of a falconer with a hooded bird of prey, perched on his gloved hand, in a forest.",
    "An image of a medieval herbalist, gathering medicinal plants in a wild, overgrown garden.",
    "An image of a dungeon deep beneath a castle, with stone walls, iron bars, and flickering torches.",
    "An image of a medieval scholar, studying ancient scrolls by candlelight in a quiet library.",
    "An image of a knight's vigil, kneeling before an altar in a chapel, awaiting his knighthood.",
    "An image of a medieval fair, with acrobats, jugglers, and food vendors entertaining a crowd.",
    "An image of a stone griffin statue, guarding the entrance to an ancient medieval keep.",
    "An image of a medieval queen, embroidering a banner with her ladies-in-waiting.",
    "An image of a hidden treasure chest, overflowing with gold coins and jewels, in a castle vault.",
    "An image of a medieval monastery, with cloistered walkways and a peaceful inner garden.",
    "An image of a knight errant, riding through a dark forest on a quest.",
    "An image of a medieval executioner, standing by a chopping block with a large axe.",
    "An image of a round table, where knights are gathered, discussing matters of importance.",
    "An image of a medieval banner, bearing a coat of arms with a lion and a sword.",
    "An image of a moat monster, lurking in the waters surrounding a dark castle.",
    "An image of a medieval princess, looking out from a high tower window.",
    "An image of a siege ram, being used to break down a castle gate.",
    "An image of a medieval apothecary, mixing potions with strange ingredients.",
    "An image of a royal court jester, performing tricks and telling jokes.",
    "An image of ancient ruins of a medieval fortress, overgrown with ivy.",
    "An image of a knight receiving a blessing from a priest before battle.",
    "An image of a medieval map, showing fantastical creatures and uncharted lands.",
    "An image of a secret passage, hidden behind a bookshelf in a castle library.",
    "An image of a medieval tournament field, prepared for contests of skill and valor.",
    "An image of a ghostly knight, haunting the corridors of an old castle."
]
positive_medieval_sentences = [add_random_commas(s) for s in raw_positive_medieval_sentences]

raw_negative_non_medieval_sentences = [ # These are now NEGATIVE NON-MEDIEVAL sentences
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
    "An image of a modern office interior, with computers, desks, and ergonomic chairs.",
    "An image of a suburban street with houses, cars parked in driveways, and manicured lawns.",
    "An image of a crowded shopping mall, with escalators, storefronts, and people carrying bags.",
    "An image of a contemporary art gallery, displaying abstract sculptures and paintings.",
    "An image of a person using a laptop computer, at a wooden desk with a coffee mug.",
    "An image of a highway interchange, with multiple lanes of traffic and overpasses.",
    "An image of a basketball game, being played in a brightly lit indoor stadium.",
    "An image of a family watching television, in a modern living room with a sofa and coffee table.",
    "An image of a construction site, with cranes, bulldozers, and workers in hard hats.",
    "An image of an airplane taking off, from a runway at a busy international airport.",
    "An image of a research scientist, in a white lab coat, working with test tubes and beakers.",
    "An image of a data center, with rows of server racks and blinking LED lights.",
    "An image of a smartphone displaying a social media feed, held in a person's hand.",
    "An image of a wind turbine farm, with large white turbines spinning against a blue sky.",
    "An image of a group of friends, taking a selfie with a smartphone at a music festival.",
    "An image of a modern kitchen, with stainless steel appliances and granite countertops.",
    "An image of a university lecture hall, with students listening to a professor.",
    "An image of a 3D printer, creating a plastic object layer by layer.",
    "An image of a satellite dish, pointed towards the sky, on the roof of a building.",
    "An image of a person jogging, on a treadmill in a brightly lit fitness gym.",
    "An image of a city park, with a playground, benches, and people relaxing on the grass.",
    "An image of a coffee barista, preparing a latte with steamed milk in a cafe.",
    "An image of a self-driving car, navigating through city traffic.",
    "An image of a video game console, with controllers and a game displayed on a large screen.",
    "An image of a solar eclipse, with the moon partially covering the sun.",
    "An image of a coral reef, teeming with colorful fish and marine life.",
    "An image of a vast desert landscape, with sand dunes stretching to the horizon.",
    "An image of a snow-capped mountain range, under a clear blue sky.",
    "An image of a dense tropical rainforest, with lush vegetation and exotic animals.",
    "An image of a volcano erupting, with lava flowing down its slopes.",
    "An image of the Northern Lights (Aurora Borealis), shimmering in the night sky.",
    "An image of a microscopic view of cells, dividing and multiplying.",
    "An image of a galaxy, with swirling stars and nebulae in deep space.",
    "An image of a plate of spaghetti bolognese, with parmesan cheese.",
    "An image of a person practicing yoga, on a mat in a serene studio.",
    "An image of a child playing with colorful building blocks on a carpeted floor.",
    "An image of a financial stock market graph, showing fluctuating prices.",
    "An image of a weather map, displaying fronts, pressure systems, and precipitation.",
    "An image of a QR code, on a product label or advertisement.",
    "An image of a group of people, wearing virtual reality headsets and interacting in a digital space."
]
negative_non_medieval_sentences = [add_random_commas(s) for s in raw_negative_non_medieval_sentences]

all_sentences = positive_medieval_sentences + negative_non_medieval_sentences
print(f"Defined {len(positive_medieval_sentences)} positive medieval sentences (with random commas) and {len(negative_non_medieval_sentences)} negative non-medieval sentences (with random commas).")

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

# Embed script's positive medieval sentences (processed with commas)
print(f"\nEmbedding {len(positive_medieval_sentences)} processed positive medieval sentences (from script)...")
script_positive_medieval_embeddings = []
for i, sentence in enumerate(positive_medieval_sentences): # positive_medieval_sentences is already processed with commas
    if (i + 1) % 10 == 0 or i == len(positive_medieval_sentences) - 1:
        print(f"  Embedding script's positive medieval sentence {i+1}/{len(positive_medieval_sentences)}: \"{sentence[:50]}...\"")
    script_positive_medieval_embeddings.append(clip_embedder.embed_text(sentence))

print(f"\nEmbedding {len(negative_non_medieval_sentences)} processed negative non-medieval sentences...")
script_negative_non_medieval_embeddings = []
for i, sentence in enumerate(negative_non_medieval_sentences): # negative_non_medieval_sentences is already processed with commas
    if (i + 1) % 10 == 0 or i == len(negative_non_medieval_sentences) - 1:
        print(f"  Embedding script's negative non-medieval sentence {i+1}/{len(negative_non_medieval_sentences)}: \"{sentence[:50]}...\"")
    script_negative_non_medieval_embeddings.append(clip_embedder.embed_text(sentence))

# --- 4. Overall Ranking Analysis ---
print(f"\n--- Overall Ranking Analysis (Each {model_name_to_analyze}.txt sentence as a model, tested against script's medieval/non-medieval examples) ---")
correct_rankings_overall = 0
total_pairs_overall = 0

# Check if all necessary embeddings lists are populated
if embeddings_from_model_txt and script_positive_medieval_embeddings and script_negative_non_medieval_embeddings:
    # Iterate through each sentence from {model_name_to_analyze}.txt (as a model)
    for i, model_theta_embedding in enumerate(embeddings_from_model_txt):
        # Iterate through each script positive medieval sentence (as a positive example)
        for j, script_positive_emb in enumerate(script_positive_medieval_embeddings): 
            score_positive_by_model = torch.dot(model_theta_embedding.squeeze(), script_positive_emb.squeeze()).item()
            
            # Iterate through each script negative non-medieval sentence (as a negative example)
            for k, script_non_positive_emb in enumerate(script_negative_non_medieval_embeddings): 
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


# --- 5. Individual Sentence Accuracy Analysis (using sentences from {model_name_to_analyze}.txt as models, tested against script's medieval/non-medieval examples) ---
print(f"\n--- Individual {model_name_to_analyze.capitalize()} Sentence Accuracies (from {model_name_to_analyze}.txt, tested against script's medieval/non-medieval examples) ---")

def calculate_single_concept_accuracy(
    emb_model_theta,  # Embedding of the target sentence from {model_name_to_analyze}.txt (acts as theta for this model)
    script_positive_medieval_embeddings_list, # List of embeddings of processed positive medieval sentences from the script
    script_negative_non_medieval_embeddings_list # List of embeddings of processed negative non-medieval sentences from the script
):
    correct_single = 0
    num_comparison_pairs = 0
    # emb_model_theta is the vector defining the current model.
    
    for emb_script_positive in script_positive_medieval_embeddings_list:
        score_positive_example_by_model = torch.dot(emb_model_theta.squeeze(), emb_script_positive.squeeze()).item()
        
        for emb_script_negative in script_negative_non_medieval_embeddings_list:
            score_negative_example_by_model = torch.dot(emb_model_theta.squeeze(), emb_script_negative.squeeze()).item()
            
            if score_positive_example_by_model > score_negative_example_by_model:
                correct_single += 1
            num_comparison_pairs += 1
            
    return correct_single / num_comparison_pairs if num_comparison_pairs > 0 else 0.0


if raw_sentences_from_model_txt and embeddings_from_model_txt and script_positive_medieval_embeddings and script_negative_non_medieval_embeddings:
    # 1. Accuracy for the first sentence from {model_name_to_analyze}.txt
    target_raw_sentence_first = raw_sentences_from_model_txt[0]
    target_emb_first = embeddings_from_model_txt[0] # This is the theta for the first model
    acc_first = calculate_single_concept_accuracy(target_emb_first, script_positive_medieval_embeddings, script_negative_non_medieval_embeddings)
    print(f"'{target_raw_sentence_first}' - accuracy: {acc_first:.4f}")

    # 2. Accuracies for the next 19 sentences from {model_name_to_analyze}.txt (indices 1 to 19)
    for i in range(1, 20): # Iterate for sentences at index 1 through 19 in {model_name_to_analyze}.txt list
        if i < len(raw_sentences_from_model_txt):
            current_raw_sentence_from_txt = raw_sentences_from_model_txt[i]
            current_emb_from_model_txt_as_model = embeddings_from_model_txt[i] # This is theta for the current model
            acc_current = calculate_single_concept_accuracy(current_emb_from_model_txt_as_model, script_positive_medieval_embeddings, script_negative_non_medieval_embeddings)
            print(f"'{current_raw_sentence_from_txt}' - accuracy: {acc_current:.4f}")
        else:
            # This means {model_name_to_analyze}.txt has fewer than (i+1) sentences.
            print(f"Warning: Index {i} out of bounds for raw_sentences_from_model_txt (length {len(raw_sentences_from_model_txt)}).")
            break # Stop if we run out of sentences
else:
    print("Skipping individual sentence accuracy calculation due to missing sentences or embeddings.")
