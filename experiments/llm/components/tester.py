# import os # Removed unused import
import torch
import numpy as np
from abc import ABC, abstractmethod
from doexpy.env.llm import create_prompt_from_tokens, DotProductModel # Removed unused create_prompt

def generate_test_sequence(rng, word_lists, horizon):
    """Generate a test sequence using the appropriate word list for each horizon step"""
    return [
        " " if rng.random() < 0.1 else rng.choice(word_lists[i])
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
    # Removed scorer_model from __init__, pass embedder to super()
    def __init__(self, embedder=None, params=None):
        super().__init__(embedder, params) # Pass embedder to base class
        print(f"Initialized {self.__class__.__name__} with {self.params}")

    @staticmethod
    def cosine_error(vec1, vec2): 
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
        # Compute cosine similarity and convert it to an error metric. 
        cos_sim = torch.nn.functional.cosine_similarity(vec1.flatten(), vec2.flatten(), dim=0)
        return 1 - cos_sim.item()

    # Added scorer_model to signature
    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__}")
        if estimator is None:
            print(f"Skipping {self.__class__.__name__}: Estimator is None.")
            return {}
        if not hasattr(estimator, 'theta_fit'):
            print(f"Skipping {self.__class__.__name__}: Estimator does not have 'theta_fit'.")
            return {}
        # Use the scorer_model passed directly to run_test
        if scorer_model is None or not hasattr(scorer_model, 'weight'):
            print(f"Skipping {self.__class__.__name__}: Scorer model passed to run_test or its weight is missing.")
            return {}

        # Get the ground-truth model weights and the estimated weights
        gt_weight = scorer_model.weight  # ground truth weight from passed scorer_model argument
        est_weight = estimator.theta_fit      # estimated weight
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
    def __init__(self, embedder=None, params=None):
        self.params = params or {}
        # embedder is passed to super() which stores it
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8) # N sequences to return
        self.use_estimator = self.params.get('use_estimator', False)
        self.beam_width = self.params.get('beam_width', 8) # Beam width for search (K in beam search)
        # Pass embedder to the base class constructor
        super().__init__(embedder=embedder, params=params)
        print(f"Initialized {self.__class__.__name__} with take_best_worst_N={self.take_best_worst_N}, "
              f"use_estimator={self.use_estimator}, beam_width={self.beam_width}")

    # Added scorer_model to signature
    def run_test(self, cfg, env, estimator, theta_star, scorer_model, training_words_list, testing_words_list, visits=None):
        """Run the image generation test using beam search.

        This tester finds the top N best and worst prompts based on the scorer model or estimator,
        using beam search to explore sequences.

        If use_estimator is True, it will use the estimator instead of the ground truth model for scoring.
        """
        # Determine which model to use for scoring and store it for helper methods
        if self.use_estimator:
            print(f"Using estimator model for {self.__class__.__name__}")
            # Store the specific estimator passed for this run
            self.current_estimator = estimator
            # No need to store scorer_model locally, it's passed to _run_beam_search_test
        # else: # scorer_model is passed directly, no need for self.current_scorer_model
            # print(f"Using ground truth model for {self.__class__.__name__}")
            # Store the specific scorer_model passed for this run
            # self.current_scorer_model = scorer_model # REMOVED

        # test_rng = np.random.RandomState(42) # Removed unused RNG
        horizon = env.max_episode_length # Use horizon from env
        # Make sure testing_words_list is a list of lists with one list per horizon step
        if not isinstance(testing_words_list[0], list):
            testing_words_list = [testing_words_list] * horizon

        # Beam search approach: find top N best and worst sequences
        # Pass the specific scorer_model for this iteration to the helper method
        return self._run_beam_search_test(cfg, env, horizon, testing_words_list, scorer_model, estimator)

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
        # Initialize beams: list of (score, sequence_tuple)
        # Start with an empty sequence tuple and score 0 (or score of base prompt if desired)
        beams = [(0.0, tuple())]

        for h in range(horizon):
            candidates = []
            # For each current beam (sequence)
            for current_score, current_sequence in beams:
                # Try appending each token from the vocabulary for this step
                for token in testing_words_list[h]:
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

    # Added scorer_model and estimator arguments
    def _run_beam_search_test(self, cfg, env, horizon, testing_words_list, scorer_model, estimator):
        """Runs beam search to find top N best and worst sequences."""
        try:
            # Determine which scoring model to use based on self.use_estimator
            if self.use_estimator:
                # Use the estimator passed to this method
                if estimator is None:
                    raise ValueError("Cannot use estimator model when estimator passed is None.")
                if self.embedder is None:
                    raise ValueError("Cannot create estimator model without an embedder instance.")
                # Use the passed estimator instance directly
                scoring_model_to_use = create_dot_product_model_from_estimator(estimator, self.embedder)
                print(f"Running beam search test with estimator model, horizon {horizon}, beam width {self.beam_width}")
            else:
                # Use the scorer_model passed to this method
                if scorer_model is None:
                    raise ValueError("Cannot use ground truth model when scorer_model passed is None.")
                scoring_model_to_use = scorer_model
                print(f"Running beam search test with ground truth model, horizon {horizon}, beam width {self.beam_width}")

            # Find N best sequences
            print("Starting beam search for best sequences...")
            # Use the determined scoring_model_to_use
            best_results = self._beam_search(env, horizon, testing_words_list, scoring_model_to_use, self.beam_width, maximize=True)
            # Extract top N best sequences and scores
            top_n_best = best_results[:self.take_best_worst_N]
            best_sequences = [list(seq) for score, seq in top_n_best] # Convert tuples back to lists
            best_scores = [score for score, seq in top_n_best]
            best_prompts = [create_prompt_from_tokens(seq, env.base_prompt) for seq in best_sequences]
            print(f"Found {len(best_sequences)} best sequences. Best score: {best_scores[0] if best_scores else 'N/A'}")

            # Find N worst sequences
            print("Starting beam search for worst sequences...")
            # Use the determined scoring_model_to_use
            worst_results = self._beam_search(env, horizon, testing_words_list, scoring_model_to_use, self.beam_width, maximize=False)
            # Extract top N worst sequences and scores (top N from the ascending sort)
            top_n_worst = worst_results[:self.take_best_worst_N]
            worst_sequences = [list(seq) for score, seq in top_n_worst] # Convert tuples back to lists
            worst_scores = [score for score, seq in top_n_worst]
            worst_prompts = [create_prompt_from_tokens(seq, env.base_prompt) for seq in worst_sequences]
            print(f"Found {len(worst_sequences)} worst sequences. Worst score: {worst_scores[0] if worst_scores else 'N/A'}")

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

