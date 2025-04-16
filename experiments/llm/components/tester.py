import os # Added import
import torch
import numpy as np
from abc import ABC, abstractmethod
from doexpy.env.llm import create_prompt_from_tokens, DotProductModel, create_prompt # Added create_prompt

def generate_test_sequence(rng, word_lists, horizon):
    """Generate a test sequence using the appropriate word list for each horizon step"""
    return [
        " " if rng.random() < 0.1 else rng.choice(word_lists[i])
        for i in range(horizon)
    ]

class BaseTester(ABC):
    """Base class for all testers with simplified interface."""
    
    def __init__(self, scorer_model=None, embedder=None, params=None): # Added embedder argument
        self.scorer_model = scorer_model
        self.embedder = embedder # Store embedder instance
        self.params = params or {}

    @abstractmethod
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list, visits=None):
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

    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__} with {self.params}")
        if estimator is None:
            print(f"Skipping {self.__class__.__name__}: Estimator is None.")
            return {}
        test_rng = np.random.RandomState(42)
        N_test_prompts = self.params['N_test_prompts']
        N_pairs_eval = self.params['N_pairs_eval']

        horizon = cfg.horizon

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
            yy, feat = self.scorer_model.score_prompt(prompt)
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
    # Updated __init__ to accept embedder and pass it to super()
    def __init__(self, scorer_model=None, embedder=None, params=None):
        super().__init__(scorer_model, embedder, params) # Pass embedder to base class
        print(f"Initialized {self.__class__.__name__} with {self.params}")
    
    @staticmethod
    def cosine_error(vec1, vec2): 
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
        # Compute cosine similarity and convert it to an error metric. 
        cos_sim = torch.nn.functional.cosine_similarity(vec1.flatten(), vec2.flatten(), dim=0)
        return 1 - cos_sim.item()

    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__}")
        if estimator is None:
            print(f"Skipping {self.__class__.__name__}: Estimator is None.")
            return {}
        if not hasattr(estimator, 'theta_fit'):
            print(f"Skipping {self.__class__.__name__}: Estimator does not have 'theta_fit'.")
            return {}
        if self.scorer_model is None or not hasattr(self.scorer_model, 'weight'):
             print(f"Skipping {self.__class__.__name__}: Scorer model or its weight is missing.")
             return {}

        # Get the ground-truth model weights and the estimated weights
        gt_weight = self.scorer_model.weight  # ground truth weight from scorer model
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
    def __init__(self, scorer_model, params=None):
        self.params = params or {}
        self.scorer_model = scorer_model
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8) if self.params else 8
        self.use_estimator = self.params.get('use_estimator', False) if self.params else False
        # Pass scorer_model to the base class constructor
        super().__init__(scorer_model=scorer_model, params=params)
        print(f"Initialized {self.__class__.__name__} with take_best_worst_N={self.take_best_worst_N}, "
              f"use_estimator={self.use_estimator}")

    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list, visits=None):
        """Run the image generation test

        This tester finds the best and worst prompts based on the scorer model or estimator.
        It builds the sequence greedily by choosing the best token at each timestep.
        
        If use_estimator is True, it will use the estimator instead of the ground truth model.
        """
        # Determine which model to use for scoring and store it for helper methods
        if self.use_estimator:
            print(f"Using estimator model for {self.__class__.__name__}")
            self.estimator = estimator  # Store for helper methods
        else:
            print(f"Using ground truth model for {self.__class__.__name__}")
            
        print(f"Running {self.__class__.__name__} with greedy approach")
        test_rng = np.random.RandomState(42)
        horizon = cfg.horizon
        
        # Make sure testing_words_list is a list of lists with one list per horizon step
        if not isinstance(testing_words_list[0], list):
            testing_words_list = [testing_words_list] * horizon
        
        # Greedy approach: build sequence by choosing best token at each step
        return self._run_greedy_test(cfg, env, test_rng, horizon, testing_words_list)
    
    def _run_greedy_test(self, cfg, env, test_rng, horizon, testing_words_list):
        """Greedy approach: build sequence by choosing best token at each step"""
        # Use the scoring model from the parent method
        try:
            if self.use_estimator:
                scoring_model = create_dot_product_model_from_estimator(self.estimator, env.embedder)
            else:
                scoring_model = self.scorer_model
                
            print(f"Running greedy test with horizon {horizon}")
            # Start with empty sequence
            best_sequence = []
            worst_sequence = []
            
            # For each position in the sequence
            for pos in range(horizon):
                print(f"Processing position {pos+1}/{horizon} with {len(testing_words_list[pos])} possible tokens")
                # Score all possible tokens at this position
                best_scores_at_pos = []
                best_tokens_at_pos = []
                worst_scores_at_pos = []
                worst_tokens_at_pos = []
                
                for token in testing_words_list[pos]:
                    # Create temporary sequence with this token
                    temp_best_sequence = best_sequence + [token]
                    temp_worst_sequence = worst_sequence + [token]
                    
                    # Pad to full horizon length if needed
                    if len(temp_best_sequence) < horizon:
                        # Use first token from each remaining position as padding
                        padding = [testing_words_list[i][0] for i in range(pos+1, horizon)]
                        temp_best_sequence = temp_best_sequence + padding
                        temp_worst_sequence = temp_worst_sequence + padding
                    
                    try:
                        # Score the sequences
                        best_prompt = create_prompt_from_tokens(temp_best_sequence, env.base_prompt)
                        best_score, _ = scoring_model.score_prompt(best_prompt)
                        
                        worst_prompt = create_prompt_from_tokens(temp_worst_sequence, env.base_prompt)
                        worst_score, _ = scoring_model.score_prompt(worst_prompt)
                        
                        best_scores_at_pos.append(best_score.item())
                        best_tokens_at_pos.append(token)
                        worst_scores_at_pos.append(worst_score.item())
                        worst_tokens_at_pos.append(token)
                    except Exception as e:
                        print(f"Error scoring token '{token}': {e}")
                        continue
                
                if not best_scores_at_pos:
                    print(f"Warning: No valid scores for position {pos+1}. Using default token.")
                    default_token = testing_words_list[pos][0]
                    best_sequence.append(default_token)
                    worst_sequence.append(default_token)
                    continue
                    
                # Choose best token for this position
                best_idx = np.argmax(best_scores_at_pos)
                best_sequence.append(best_tokens_at_pos[best_idx])
                
                # Choose worst token for this position
                worst_idx = np.argmin(worst_scores_at_pos)
                worst_sequence.append(worst_tokens_at_pos[worst_idx])
                
                print(f"Position {pos+1}: Best token '{best_tokens_at_pos[best_idx]}' (score: {best_scores_at_pos[best_idx]:.4f}), "
                      f"Worst token '{worst_tokens_at_pos[worst_idx]}' (score: {worst_scores_at_pos[worst_idx]:.4f})")
        except Exception as e:
            print(f"Error in greedy test: {e}")
            return {
                "image_generation": {
                    "best_prompts": [],
                    "best_scores": [],
                    "worst_prompts": [],
                    "worst_scores": [],
                    "best_sequence": [],
                    "worst_sequence": []
                },
                "best_image_score": 0,
                "worst_image_score": 0,
                "avg_top_image_score": 0,
                "error": str(e)
            }
        
        # For the final position, get the top N best and worst completions
        if horizon > 0:
            print(f"Getting top {self.take_best_worst_N} best and worst completions for the final position")
            # Score all possible completions for the last position
            final_pos = horizon - 1
            all_scores = []
            all_prompts = []
            
            # Use best_sequence up to the second-to-last position
            best_prefix = best_sequence[:-1]
            worst_prefix = worst_sequence[:-1]
            
            # Try all tokens for the last position
            for token in testing_words_list[final_pos]:
                # Best sequence with this final token
                full_best_sequence = best_prefix + [token]
                best_prompt = create_prompt_from_tokens(full_best_sequence, env.base_prompt)
                best_score, _ = scoring_model.score_prompt(best_prompt)
                
                # Worst sequence with this final token
                full_worst_sequence = worst_prefix + [token]
                worst_prompt = create_prompt_from_tokens(full_worst_sequence, env.base_prompt)
                worst_score, _ = scoring_model.score_prompt(worst_prompt)
                
                all_scores.append((best_score.item(), best_prompt, "best"))
                all_scores.append((worst_score.item(), worst_prompt, "worst"))
            
            # Sort all scores
            all_scores.sort(reverse=True)  # Sort by score descending
            
            # Get top N best prompts
            best_prompts = []
            best_scores = []
            for i in range(min(self.take_best_worst_N, len(all_scores))):
                if all_scores[i][2] == "best":
                    best_prompts.append(all_scores[i][1])
                    best_scores.append(all_scores[i][0])
                if len(best_prompts) >= self.take_best_worst_N:
                    break
            
            print(f"Found {len(best_prompts)} best prompts with scores ranging from {max(best_scores):.4f} to {min(best_scores) if best_scores else 0:.4f}")
            
            # Sort all scores in ascending order for worst
            all_scores.sort()  # Sort by score ascending
            
            # Get top N worst prompts
            worst_prompts = []
            worst_scores = []
            for i in range(min(self.take_best_worst_N, len(all_scores))):
                if all_scores[i][2] == "worst":
                    worst_prompts.append(all_scores[i][1])
                    worst_scores.append(all_scores[i][0])
                if len(worst_prompts) >= self.take_best_worst_N:
                    break
            
            print(f"Found {len(worst_prompts)} worst prompts with scores ranging from {max(worst_scores) if worst_scores else 0:.4f} to {min(worst_scores) if worst_scores else 0:.4f}")
        else:
            # Handle edge case of horizon=0
            best_prompts = []
            best_scores = []
            worst_prompts = []
            worst_scores = []
        
        return {
            "image_generation": {
                "best_prompts": best_prompts,
                "best_scores": best_scores,
                "worst_prompts": worst_prompts,
                "worst_scores": worst_scores,
                "best_sequence": best_sequence,
                "worst_sequence": worst_sequence
            },
            "best_image_score": best_scores[0] if best_scores else 0,
            "worst_image_score": worst_scores[0] if worst_scores else 0,
            "avg_top_image_score": sum(best_scores) / len(best_scores) if best_scores else 0
        }


class VisitsTester(BaseTester):
    """
    Inspects visit trajectories by printing actions and converting them to prompts.
    Can load visits from a file if not provided directly.
    """
    def __init__(self, scorer_model=None, embedder=None, params=None, env=None): # Added env
        super().__init__(scorer_model, embedder, params)
        self.env = env # Store env for create_prompt
        self.output_filename = self.params.get('output_filename', None) # Optional output file
        print(f"Initialized {self.__class__.__name__} with params: {self.params}")
        if self.env is None:
             # This tester absolutely needs the env to create prompts
             raise ValueError(f"{self.__class__.__name__} requires the 'env' object for prompt creation.")

    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list, visits=None):
        print(f"Running {self.__class__.__name__}")

        loaded_visits = None
        # Prioritize visits passed directly
        if visits:
            print("Using visits provided by the experiment run.")
            loaded_visits = visits
        # If not passed, try loading from path specified in main config
        elif cfg.get('visits_path'):
            visits_path = cfg.visits_path
            print(f"Attempting to load visits from: {visits_path}")
            if os.path.exists(visits_path):
                try:
                    loaded_visits = torch.load(visits_path)
                    print(f"Successfully loaded visits from {visits_path}")
                except Exception as e:
                    print(f"Error loading visits from {visits_path}: {e}")
                    return {"visits_tester_error": f"Failed to load {visits_path}"}
            else:
                print(f"Warning: Visits path specified but not found: {visits_path}")
                return {"visits_tester_error": f"Visits file not found: {visits_path}"}
        else:
            print("Warning: No visits provided directly or via visits_path.")
            return {"visits_tester_warning": "No visits data available"}

        if not loaded_visits:
             print("Error: Visits data is empty or failed to load.")
             return {"visits_tester_error": "Visits data is empty or failed to load"}

        # --- Process and Print Visits/Prompts ---
        output_lines = []
        try:
            num_policies = len(loaded_visits)
            num_episodes = len(loaded_visits[0]) if num_policies > 0 else 0
            print(f"Processing {num_policies} policies and {num_episodes} episodes.")
            output_lines.append(f"--- Visits Inspection ---")
            output_lines.append(f"Number of Policies: {num_policies}")
            output_lines.append(f"Number of Episodes: {num_episodes}")
            output_lines.append("-" * 25)

            for p_idx in range(num_policies):
                output_lines.append(f"\nPolicy {p_idx + 1}:")
                for ep_idx in range(num_episodes):
                    try:
                        # visits[policy_idx][ep_idx] = (states, actions)
                        actions = loaded_visits[p_idx][ep_idx][1]
                        if isinstance(actions, torch.Tensor):
                            actions = actions.cpu().numpy()
                        actions = list(map(int, actions)) # Ensure list of ints

                        # Use the env stored during init
                        prompt = create_prompt(actions, self.env)

                        output_lines.append(f"  Episode {ep_idx + 1}:")
                        output_lines.append(f"    Actions: {actions}")
                        output_lines.append(f"    Prompt : '{prompt}'")

                    except IndexError:
                        output_lines.append(f"  Episode {ep_idx + 1}: Error - Missing data")
                    except Exception as e:
                         output_lines.append(f"  Episode {ep_idx + 1}: Error - Processing failed: {e}")

        except Exception as e:
            print(f"Error processing visits: {e}")
            output_lines.append(f"\nError during processing: {e}")
            return {"visits_tester_error": f"Processing failed: {e}"}

        # Print to console
        print("\n".join(output_lines))

        # Save to file if filename is provided
        if self.output_filename:
            # Construct path using BaseSaver's logic (needs results_dir, experiment_id)
            # We need results_dir and experiment_id from the main experiment context
            # Let's assume they are available in cfg for now, or pass them?
            # For simplicity, let's just save relative to the current run's results_dir
            # We need a way to get results_dir here. Let's pass it via cfg for now.
            # This assumes the tester is instantiated within the experiment context.
            # A cleaner way might be needed if used standalone.

            # We need results_dir and experiment_id. Let's try getting them from cfg.
            results_dir = cfg.get("results_dir", ".") # Default to current dir if not found
            experiment_id = cfg.get("experiment_id", None)

            # Build filename with experiment_id if provided
            name, ext = os.path.splitext(self.output_filename)
            if experiment_id:
                 filename = f"{name}-{experiment_id}{ext}"
            else:
                 filename = self.output_filename

            output_path = os.path.join(results_dir, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure dir exists

            try:
                with open(output_path, 'w') as f:
                    f.write("\n".join(output_lines))
                print(f"Saved visit prompts to: {output_path}")
            except Exception as e:
                print(f"Error saving visit prompts to {output_path}: {e}")
                return {"visits_tester_error": f"Failed to save prompts file: {e}"}

        return {} # Return empty metrics dict as this is for inspection

