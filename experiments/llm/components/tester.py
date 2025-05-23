import os # Added os import
import torch
import numpy as np
from abc import ABC, abstractmethod
from doexpy.env.llm import create_prompt_from_tokens, DotProductModel # Removed unused create_prompt

def generate_test_sequence(rng, word_lists, horizon):
    """Generate a test sequence using the appropriate word list for each horizon step"""
    return [
        " " if rng.random() < 0.3 else rng.choice(word_lists[i])
        for i in range(horizon)
    ]

class BaseTester(ABC):
    """Base class for all testers with simplified interface."""

    # Accept env, embedder, params
    def __init__(self, env=None, embedder=None, params=None):
        self.env = env # Store env instance
        self.embedder = embedder # Store embedder instance
        self.params = params or {}

    @abstractmethod
    # Added scorer_model argument to run_test signature
    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        """
        Run tests and return metrics dictionary.

        Args:
            cfg: Hydra configuration object.
            env: Environment instance.
            estimator: Fitted estimator instance (can be None).
            theta_star: Ground truth scoring function.
            training_words_list: List of lists of training words per horizon step.
            testing_words_list: List of lists of testing words per horizon step.
            visits: Optional list of visit trajectories from the experiment run.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

class PreferenceTester(BaseTester):
    """Tests preference prediction accuracy on held-out test data."""

    # Added scorer_model to signature
    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__} with {self.params}")
        if estimator is None:
            print(f"Skipping {self.__class__.__name__}: Estimator is None.")
            return {}
        test_rng = np.random.RandomState(42)
        N_test_prompts = self.params['N_test_prompts']
        N_pairs_eval = self.params['N_pairs_eval']

        horizon = env.max_episode_length # Use horizon from env

        # Make sure testing_words_list is a list of lists with one list per horizon step
        assert isinstance(testing_words_list[0], list)

        print(f"Generating {N_test_prompts} test sequences with horizon {horizon}")
        test_sequences = [generate_test_sequence(test_rng, testing_words_list, horizon)
                for _ in range(N_test_prompts)]

        print(f"Computing embeddings and predictions for {len(test_sequences)} sequences")
        # Compute embeddings and predictions
        xtest = []
        ytest = []
        for sequence in test_sequences:
            # Create prompt directly from the sequence tokens
            prompt = create_prompt_from_tokens(sequence, env.base_prompt)
            # Use the passed scorer_model instance for ground truth scoring
            # This was passed as an argument to run_test, use it directly
            # Need to access scorer_model passed to run_test, not self.scorer_model
            # Assuming scorer_model is passed correctly to run_test
            # We need to add scorer_model to the arguments list if it's not there
            # Let's assume it *is* passed as an argument for now, based on previous context.
            # If not, the code will fail later with NameError.
            # Re-checking the method signature: it's not passed. This needs fixing.
            # Use the scorer_model passed directly to run_test
            if scorer_model is None: # Check the passed argument
                print(f"Skipping sequence in {self.__class__.__name__}: scorer_model passed to run_test is None.")
                continue # Skip if the specific scorer model is missing
            yy, feat = scorer_model.score_prompt(prompt) # Use the argument scorer_model
            xtest.append(feat.detach().cpu())
            ytest.append(yy.detach().cpu())

        xtest = torch.vstack(xtest)
        ytest = torch.vstack(ytest)
        ypred = estimator.mean(xtest)

        print(f"Evaluating {N_pairs_eval} preference pairs")
        pair_indices = np.array([(i, j) for i in range(N_test_prompts) for j in range(i+1, N_test_prompts)])
        selected_pairs = pair_indices[test_rng.choice(len(pair_indices), N_pairs_eval, replace=False)]

        correct_preferences = 0
        for i, j in selected_pairs:
            gt_prefers_i = (ytest[i] >= ytest[j]).item()
            pred_prefers_i = (ypred[i] >= ypred[j]).item()
            if gt_prefers_i == pred_prefers_i:
                correct_preferences += 1

        error = 1.0 - (correct_preferences / N_pairs_eval)

        print(f"Preference error: {error:.4f} ({correct_preferences}/{N_pairs_eval} correct)")

        return {"preference_error": error}

class CosineTester(BaseTester):
    # Accept env, embedder, params like BaseTester
    def __init__(self, env=None, embedder=None, params=None):
        # Pass all relevant arguments to the base class constructor
        super().__init__(env=env, embedder=embedder, params=params)
        print(f"Initialized {self.__class__.__name__} with {self.params}")

    @staticmethod
    def cosine_error(vec1, vec2): 
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
        # Compute cosine similarity and convert it to an error metric. 
        cos_sim = torch.nn.functional.cosine_similarity(vec1.flatten(), vec2.flatten(), dim=0)
        return 1 - cos_sim.item()

    # Added scorer_model and feedback to signature
    def run_test(self, cfg, env, estimator, feedback, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__}")

        if feedback is None:
            print(f"Skipping {self.__class__.__name__}: Feedback object is None.")
            return {}

        est_weight = feedback.get_learned_theta()
        if est_weight is None:
            print(f"Skipping {self.__class__.__name__}: Learned theta not available from feedback object.")
            return {}

        # Use the scorer_model passed directly to run_test
        if scorer_model is None or not hasattr(scorer_model, 'weight'):
            print(f"Skipping {self.__class__.__name__}: Scorer model passed to run_test or its weight is missing.")
            return {}

        # Get the ground-truth model weights
        gt_weight = scorer_model.weight  # ground truth weight from passed scorer_model argument
        
        error = self.cosine_error(est_weight, gt_weight)
        print(f"Cosine error: {error:.4f}")
        return {"cosine_error": error}

def create_dot_product_model_from_estimator(estimator, embedder):
    """
    Convert a RegularizedMultinomialEstimator to a DotProductModel
    
    Args:
        estimator: RegularizedMultinomialEstimator instance
        embedder: CLIPEmbedder instance
        
    Returns:
        DotProductModel that can be used like the ground truth model
    """
    # Extract the weight from the estimator
    weight = estimator.theta_ml()
    
    # Create a DotProductModel with the weight
    # The DotProductModel constructor will handle the shape standardization
    try:
        model = DotProductModel(embedder, weight)
        return model
    except ValueError as e:
        raise ValueError(f"Failed to create DotProductModel from estimator: {e}")

class ImageGenerationTester(BaseTester):
    # Removed scorer_model from __init__, pass embedder to super()
    def __init__(self, env=None, embedder=None, params=None):
        self.params = params or {}
        # embedder is passed to super() which stores it
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8) # N sequences to return
        self.prompt_ranking_model = self.params.get('prompt_ranking_model', 'gt') # 'gt' or scorer_model name
        self.beam_width = self.params.get('beam_width', 15) # Beam width for search (K in beam search)
        # Pass env, embedder to the base class constructor
        super().__init__(env=env, embedder=embedder, params=params)
        print(f"Initialized {self.__class__.__name__} with take_best_worst_N={self.take_best_worst_N}, "
              f"prompt_ranking_model='{self.prompt_ranking_model}', beam_width={self.beam_width}")

    # Added scorer_model to signature
    # Added all_estimators and all_gt_scorer_models for specific model selection
    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None, all_estimators=None, all_gt_scorer_models=None):
        """Run the image generation test using beam search.

        This tester finds the top N best and worst prompts based on the scorer model or estimator,
        using beam search to explore sequences.

        The model for ranking is determined by `self.prompt_ranking_model`.
        """
        # test_rng = np.random.RandomState(42) # Removed unused RNG
        horizon = env.max_episode_length # Use horizon from env
        # Make sure testing_words_list is a list of lists with one list per horizon step
        if not isinstance(testing_words_list[0], list):
            testing_words_list = [testing_words_list] * horizon

        # Beam search approach: find top N best and worst sequences
        # Pass necessary models and lists to the helper method
        return self._run_beam_search_test(cfg, env, horizon, testing_words_list,
                                          scorer_model, estimator, # These are for the current (first) model iteration
                                          all_estimators, all_gt_scorer_models)

    def _score_sequence(self, sequence, env, horizon, testing_words_list, scoring_model):
        """Scores a potentially partial sequence by padding and using the scoring model."""
        # Pad sequence if it's shorter than the horizon
        padded_sequence = list(sequence) # Make a mutable copy
        current_len = len(padded_sequence)
        if current_len < horizon:
            # Use first token from each remaining position's vocab as padding
            padding = [testing_words_list[i][0] for i in range(current_len, horizon)]
            padded_sequence.extend(padding)

        # Create prompt and score
        prompt = create_prompt_from_tokens(padded_sequence, env.base_prompt)
        score, _ = scoring_model.score_prompt(prompt)
        return score.item()

    def _beam_search(self, env, horizon, testing_words_list, scoring_model, beam_width, maximize):
        """Performs beam search to find sequences optimizing the score."""
        print(f"ImageGenerationTester._beam_search: Received testing_words_list with {len(testing_words_list)} steps for horizon {horizon}.")
        if testing_words_list and len(testing_words_list) == horizon:
            for i_h, h_vocab in enumerate(testing_words_list):
                print(f"  Vocabulary for step {i_h}: {len(h_vocab)} words. First few: {h_vocab[:3] if h_vocab else '[]'}")
        elif len(testing_words_list) != horizon :
            print(f"  Warning: testing_words_list length ({len(testing_words_list)}) does not match horizon ({horizon}).")
        else: # testing_words_list is empty
            print("  Received empty testing_words_list.")

        # Initialize beams: list of (score, sequence_tuple)
        # Start with an empty sequence tuple and score 0 (or score of base prompt if desired)
        beams = [(0.0, tuple())]

        for h in range(horizon):
            if h >= len(testing_words_list) or not testing_words_list[h]:
                print(f"Warning: ImageGenerationTester._beam_search: testing_words_list[{h}] is empty or not available. Beam search cannot proceed for this step.")
                beams = [] # Clear beams as no new candidates can be formed
                break # Terminate beam search

            candidates = []
            # For each current beam (sequence)
            for current_score, current_sequence in beams:
                # Try appending each token from the vocabulary for this step
                for token in testing_words_list[h]: # This is now safe due to the check above
                    new_sequence = current_sequence + (token,)
                    try:
                        # Score the new (potentially partial) sequence
                        score = self._score_sequence(new_sequence, env, horizon, testing_words_list, scoring_model)
                        candidates.append((score, new_sequence))
                    except Exception as e:
                        print(f"Error scoring sequence {new_sequence} with token '{token}': {e}")
                        continue # Skip this candidate if scoring fails

            # Sort candidates by score (descending for maximize, ascending for minimize)
            candidates.sort(key=lambda x: x[0], reverse=maximize)

            # Select top beam_width candidates as the new beams
            beams = candidates[:beam_width]

            if not beams: # Stop if no valid candidates were found
                print(f"Warning: Beam search terminated early at step {h+1} due to no valid candidates.")
                break

        # Final beams are sorted by score according to 'maximize'
        return beams # Returns list of (score, sequence_tuple)

    # Added scorer_model (first GT model), estimator (first learned estimator),
    # all_estimators_list, and all_gt_scorer_models_list
    def _run_beam_search_test(self, cfg, env, horizon, testing_words_list,
                              first_gt_scorer_model, first_estimator,
                              all_estimators_list, all_gt_scorer_models_list):
        """Runs beam search to find top N best and worst sequences."""
        from omegaconf import OmegaConf # For accessing list config

        try:
            scoring_model_for_ranking = None
            if self.prompt_ranking_model == 'gt':
                if first_gt_scorer_model is None:
                    # Try to get the name of the first model for a more informative error
                    first_model_name_in_exp = OmegaConf.to_container(cfg.experiment.scorer_model_names, resolve=True)[0] if OmegaConf.is_list(cfg.experiment.scorer_model_names) and len(cfg.experiment.scorer_model_names)>0 else "N/A"
                    raise ValueError(f"Cannot use 'gt' for prompt ranking: Ground truth scorer model for '{first_model_name_in_exp}' is not available.")
                scoring_model_for_ranking = first_gt_scorer_model
                # Log which GT model is being used (name of the first model in the experiment config)
                first_model_name_in_exp = OmegaConf.to_container(cfg.experiment.scorer_model_names, resolve=True)[0] if OmegaConf.is_list(cfg.experiment.scorer_model_names) and len(cfg.experiment.scorer_model_names)>0 else "N/A"
                print(f"ImageGenerationTester: Using ground truth model ('{first_model_name_in_exp}') for prompt ranking. Horizon {horizon}, beam width {self.beam_width}")
            else: # It's a specific model name
                model_name_to_use = self.prompt_ranking_model
                
                # Get experiment's model names list from cfg
                experiment_model_names = OmegaConf.to_container(cfg.experiment.scorer_model_names, resolve=True)
                if not experiment_model_names or not isinstance(experiment_model_names, list):
                    raise ValueError(f"Cannot rank with '{model_name_to_use}': No scorer_model_names list configured in experiment.")

                if model_name_to_use not in experiment_model_names:
                    raise ValueError(f"Prompt ranking model '{model_name_to_use}' not found in experiment's scorer_model list: {experiment_model_names}")
                
                model_idx = experiment_model_names.index(model_name_to_use)
                
                if all_estimators_list is None or model_idx >= len(all_estimators_list) or all_estimators_list[model_idx] is None:
                    raise ValueError(f"Estimator for prompt ranking model '{model_name_to_use}' (index {model_idx}) is not available from all_estimators_list.")
                
                target_estimator = all_estimators_list[model_idx]
                if self.embedder is None: # self.embedder is from BaseTester.__init__
                    raise ValueError(f"Cannot create estimator model for prompt ranking ('{model_name_to_use}') without an embedder instance.")
                scoring_model_for_ranking = create_dot_product_model_from_estimator(target_estimator, self.embedder)
                print(f"ImageGenerationTester: Using estimator for '{model_name_to_use}' for prompt ranking. Horizon {horizon}, beam width {self.beam_width}")

            # Find N best sequences
            print("Starting beam search for best sequences...")
            best_results = self._beam_search(env, horizon, testing_words_list, scoring_model_for_ranking, self.beam_width, maximize=True)
            # Extract top N best sequences and scores
            top_n_best = best_results[:self.take_best_worst_N]
            best_sequences = [list(seq) for score, seq in top_n_best] # Convert tuples back to lists
            best_scores = [score for score, seq in top_n_best]
            best_prompts = [create_prompt_from_tokens(seq, env.base_prompt) for seq in best_sequences]
            print(f"Found {len(best_sequences)} best sequences. Best score: {best_scores[0] if best_scores else 'N/A'}")

            # Find N worst sequences
            print("Starting beam search for worst sequences...")
            worst_results = self._beam_search(env, horizon, testing_words_list, scoring_model_for_ranking, self.beam_width, maximize=False)
            # Extract top N worst sequences and scores (top N from the ascending sort)
            top_n_worst = worst_results[:self.take_best_worst_N]
            worst_sequences = [list(seq) for score, seq in top_n_worst] # Convert tuples back to lists
            worst_scores = [score for score, seq in top_n_worst]
            worst_prompts = [create_prompt_from_tokens(seq, env.base_prompt) for seq in worst_sequences]
            print(f"Found {len(worst_sequences)} worst sequences. Worst score: {worst_scores[0] if worst_scores else 'N/A'}")

            # Log the number of prompts found before returning
            print(f"ImageGenerationTester: Beam search complete. Found {len(best_prompts)} best prompts and {len(worst_prompts)} worst prompts.")

        except Exception as e:
            print(f"Error during beam search test: {e}")
            # Return an error structure
            return {
                "image_generation": {
                    "best_prompts": [], "best_scores": [], "best_sequences": [],
                    "worst_prompts": [], "worst_scores": [], "worst_sequences": []
                },
                "best_image_score": 0, "worst_image_score": 0, "avg_top_image_score": 0,
                "error": str(e)
            }

        # Return results in the expected format
        return {
            "image_generation": {
                "best_prompts": best_prompts,
                "best_scores": best_scores,
                "best_sequences": best_sequences, # Add sequences
                "worst_prompts": worst_prompts,
                "worst_scores": worst_scores,
                "worst_sequences": worst_sequences # Add sequences
            },
            "best_image_score": best_scores[0] if best_scores else 0,
            "worst_image_score": worst_scores[0] if worst_scores else 0,
            # Avg score of the N best sequences found by beam search
            "avg_top_image_score": sum(best_scores) / len(best_scores) if best_scores else 0
        }

class HumanFeedbackBenchmarkTester(BaseTester):
    """
    Tests a human-trained estimator on benchmark episodes defined in a feedback.json file.
    Compares estimator predictions against a ground truth scorer model.
    The environment's base_prompt is expected to be set to the 'user_prompt'
    from the feedback.json file before this tester is run by LLMExperiment.
    Visits data is also expected to be loaded by LLMExperiment and passed to run_test.
    """
    def __init__(self, env=None, embedder=None, params=None):
        super().__init__(env=env, embedder=embedder, params=params)
        # No specific params needed for this tester in its config,
        # it relies on cfg.feedback_path, cfg.algorithm, and passed objects.
        print(f"Initialized {self.__class__.__name__}")

    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        # For this benchmark, theta_star and scorer_model (ground truth) are not used to determine correctness.
        # Correctness is based on matching the human's recorded preference.
        print(f"Running {self.__class__.__name__} (Comparing estimator to human choices on benchmark episodes)")

        if estimator is None:
            print(f"  Skipping {self.__class__.__name__}: Estimator (human-trained) is None.")
            return {}
        if visits is None:
            print(f"  Skipping {self.__class__.__name__}: visits data is None (must be loaded by LLMExperiment).")
            return {}
        
        # Ensure feedback_path is available in cfg
        feedback_path_str = getattr(cfg, 'feedback_path', None)
        if not feedback_path_str:
            print(f"  Skipping {self.__class__.__name__}: feedback_path not found in configuration.")
            return {}
        
        # Resolve feedback_path using Hydra's utility
        from hydra.utils import to_absolute_path
        abs_feedback_path = to_absolute_path(feedback_path_str)
        if not os.path.exists(abs_feedback_path):
            print(f"  Skipping {self.__class__.__name__}: feedback_path file not found: {abs_feedback_path}")
            return {}

        # Load feedback.json to get benchmark_episode_keys and human preferences
        try:
            import json 
            import re 
            from doexpy.env.llm import create_prompt # For creating prompts

            with open(abs_feedback_path, 'r') as f:
                feedback_data = json.load(f)
            
            benchmark_episode_keys = feedback_data.get("benchmark_episode_keys", [])
            human_preferences_list = feedback_data.get("preferences", [])

            if not benchmark_episode_keys:
                print(f"  Warning: No 'benchmark_episode_keys' found in {abs_feedback_path}. Nothing to test.")
                return {"benchmark_accuracy": 0.0, "benchmark_comparisons_count": 0}
            if not human_preferences_list:
                print(f"  Warning: No 'preferences' list found in {abs_feedback_path}. Cannot determine human choices for benchmark.")
                return {"benchmark_accuracy": 0.0, "benchmark_comparisons_count": 0}
        except Exception as e:
            print(f"  Error loading or parsing feedback_path {abs_feedback_path}: {e}")
            return {}

        print(f"  Base prompt for benchmark evaluation (from env): '{env.base_prompt}'")
        print(f"  Evaluating on {len(benchmark_episode_keys)} benchmark episode keys using human feedback as ground truth.")

        correct_predictions = 0
        total_comparisons = 0
        
        episode_key_pattern = re.compile(r"([a-zA-Z0-9_]+)-(\d+)") # For "alg-ep_idx"
        # Filename pattern from feedback.json: "images/alg-design_episode_000_timestep_01.png"
        feedback_filename_pattern = re.compile(r"images/alg-([a-zA-Z0-9_]+)_episode_(\d+)_timestep_(\d+)\.png$")

        # Create a lookup for human preferences: (alg_name, ep_idx_str, ts_str) -> human_choice_1_based
        human_choices_lookup = {}
        for pref_entry in human_preferences_list:
            filename = pref_entry.get("filename", "")
            fn_match = feedback_filename_pattern.search(filename)
            if fn_match:
                alg, ep_str, ts_str = fn_match.groups()
                human_choices_lookup[(alg, ep_str, ts_str)] = pref_entry.get("preference")
        
        num_policies_in_visits = len(visits)
        if num_policies_in_visits == 0:
            print("    Error: Visits data is empty (no policies). Cannot proceed with benchmark.")
            return {"benchmark_accuracy": 0.0, "benchmark_comparisons_count": 0, "error": "Empty visits data"}

        for episode_key in benchmark_episode_keys:
            ep_key_match = episode_key_pattern.match(episode_key)
            if not ep_key_match:
                print(f"    Warning: Could not parse benchmark episode key: {episode_key}")
                continue
            
            alg_name_from_key = ep_key_match.group(1)
            ep_idx_from_key = int(ep_key_match.group(2)) # This is the 0-based episode index from the key

            if alg_name_from_key.lower() != cfg.algorithm.lower():
                print(f"    Critical Error: Benchmark episode key '{episode_key}' (alg: {alg_name_from_key}) "
                      f"does not match current experiment algorithm '{cfg.algorithm}'. Skipping this episode.")
                continue

            for h_prefix_len in range(1, env.max_episode_length + 1):
                # Key for human_choices_lookup: (alg_name, "000"-padded ep_idx, "00"-padded ts)
                lookup_key = (alg_name_from_key, f"{ep_idx_from_key:03d}", f"{h_prefix_len:02d}")
                human_preferred_policy_1_based = human_choices_lookup.get(lookup_key)

                if human_preferred_policy_1_based is None:
                    continue # No human feedback for this specific comparison point
                
                human_preferred_policy_0_based = human_preferred_policy_1_based - 1

                comparison_embeddings_list = []
                valid_comparison_point = True

                for policy_idx in range(num_policies_in_visits):
                    try:
                        actions_for_policy_raw = visits[policy_idx][ep_idx_from_key][1]
                        actions_for_policy = list(map(int, actions_for_policy_raw.cpu().tolist() if isinstance(actions_for_policy_raw, torch.Tensor) else actions_for_policy_raw))

                        if h_prefix_len > len(actions_for_policy):
                            valid_comparison_point = False; break
                        
                        truncated_actions = actions_for_policy[:h_prefix_len]
                        prompt = create_prompt(truncated_actions, env)
                        embedding = self.embedder.embed_text(prompt)
                        comparison_embeddings_list.append(embedding.detach().cpu())
                    except IndexError:
                        valid_comparison_point = False; break
                    except Exception as e:
                        print(f"    Error processing benchmark data for policy {policy_idx}, ep {ep_idx_from_key}, h {h_prefix_len}: {e}")
                        valid_comparison_point = False; break
                
                if not valid_comparison_point or not comparison_embeddings_list or len(comparison_embeddings_list) != num_policies_in_visits:
                    continue

                comparison_embeddings_tensor = torch.cat(comparison_embeddings_list, dim=0)
                reshaped_embeddings = comparison_embeddings_tensor.unsqueeze(0)
                
                predicted_probs = estimator.predict_proba(reshaped_embeddings)
                estimator_predicted_policy_0_based = torch.argmax(predicted_probs.squeeze()).item()

                if estimator_predicted_policy_0_based == human_preferred_policy_0_based:
                    correct_predictions += 1
                total_comparisons += 1
        
        benchmark_accuracy = (correct_predictions / total_comparisons) if total_comparisons > 0 else 0.0
        print(f"  Human Feedback Benchmark Accuracy: {benchmark_accuracy:.4f} ({correct_predictions}/{total_comparisons} comparisons)")
        return {"benchmark_accuracy": benchmark_accuracy, "benchmark_comparisons_count": total_comparisons}

