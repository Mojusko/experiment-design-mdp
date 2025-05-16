import torch
import torch.nn.functional as F
import numpy as np
from doexpy.env.llm import create_prompt, create_prompt_from_tokens

from doexpy.functionals.doe_static_functionals import (
    DesignA, DesignD, MultiPolicyOrigDesignA, MultiPolicyOrigDesignD, MultiPolicyOrigDesignC
)
from doexpy.functionals.doe_adaptive_functionals import (
    AdaptiveOrigDesignA, AdaptiveOrigDesignC # Removed AdaptiveOrigDesignD
)
# from doexpy.feedback.feedback_base import EmptyFeedback # Removed EmptyFeedback
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.regression.kernelized_features import KernelizedFeatures
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
from stpy.probability.multinomial_likelihood import MultinomialLikelihood
import warnings
from stpy.regularization.regularizer import L2Regularizer
from abc import abstractmethod

class BaseFeedback:
    def __init__(self, env, design, estimator):
        self.env = env
        self.design = design
        self.estimator = estimator
        self._collected_data = []
    
    @abstractmethod
    def collect_labels(self, cfg, new_visits, theta_star):
        pass

    def fit_estimator(self):
        pass

class NumericalFeedback(BaseFeedback):
    def collect_labels(self, cfg, new_visits, theta_star):
        # Use horizon from the environment
        horizon = self.env.max_episode_length
        prefix_range = range(1, horizon+1) if cfg.dense_feedback else range(horizon, horizon+1)
        num_episodes = len(new_visits)

        x_list, y_list = [], []
        for ep in range(num_episodes):
            actions = new_visits[ep][1]
            for prefix_len in prefix_range:
                truncated = actions[:prefix_len]
                yy, xx = theta_star(truncated)
                x_list.append(xx.detach().cpu())
                y_list.append(yy.detach().cpu())
        
        x_torch = torch.vstack(x_list)
        y_torch = torch.vstack(y_list)
        self._collected_data.append((x_torch, y_torch))

    def fit_estimator(self, preloaded_theta=None):
        """
        Fit the estimator with collected data or preloaded theta
        
        Args:
            preloaded_theta: Pre-computed theta parameter to use instead of fitting
        """
        if preloaded_theta is not None:
            # Use preloaded theta directly
            if hasattr(self.estimator, 'set_theta'):
                self.estimator.set_theta(preloaded_theta)
            else:
                # If the estimator doesn't have set_theta, try to set the attribute
                self.estimator.theta_fit = preloaded_theta
                self.estimator.fitted = True
            return
            
        if not self._collected_data:
            return
            
        # Combine all collected x and y data
        x_all = torch.vstack([x for x, _ in self._collected_data])
        y_all = torch.vstack([y for _, y in self._collected_data])
        self.estimator.load_data((x_all, y_all))
        self.estimator.fit()

class MultinomialFeedback(BaseFeedback):
    def collect_labels(self, cfg, new_visits, theta_star):
        # Use horizon from the environment
        horizon = self.env.max_episode_length
        num_policies = cfg.feedback.num_policies
        num_episodes = len(new_visits[0])
        prefix_range = range(1, horizon+1) if cfg.dense_feedback else range(horizon, horizon+1)
        num_samples = num_episodes * len(prefix_range) # Correct calculation based on prefix_range length
        embedding_dim = self.env.get_dim() # Get embedding dimension from env
        print(f"Collect Labels: num_episodes = {num_episodes}, horizon = {horizon}, dense={cfg.dense_feedback}, expected_samples = {num_samples}") # DEBUG PRINT

        # Store collected embeddings and labels directly
        collected_embeddings = []
        collected_labels = []

        for ep in range(num_episodes):
            policy_actions = [visits[ep][1] for visits in new_visits]
            for prefix_len in prefix_range:
                # Get truncated actions for the current prefix length
                trunc_actions = [act[:prefix_len] for act in policy_actions]

                # Get scores AND embeddings for each policy's truncated action sequence
                scores = []
                embeddings = []
                for ta in trunc_actions:
                    score, embedding = theta_star(ta) # Get both score and embedding
                    scores.append(score)
                    # Ensure embedding is on CPU and detached for storage
                    embeddings.append(embedding.detach().cpu())

                # Stack scores and embeddings for this comparison
                vals = torch.cat(scores, dim=0) # Shape [num_policies]
                # Stack embeddings: list of [1, dim] -> [num_policies, dim]
                comparison_embeddings_tensor = torch.cat(embeddings, dim=0)

                # Generate multinomial label based on scores
                # Squeeze vals to make it 1D [num_policies] before softmax
                probs = F.softmax(vals.squeeze().detach(), dim=0)
                label_idx = torch.multinomial(probs, 1).item() # Get the index as an integer

                # Create one-hot label tensor
                label_tensor = torch.zeros(num_policies)
                label_tensor[label_idx] = 1

                # Store the embeddings tensor and the label tensor for this sample
                collected_embeddings.append(comparison_embeddings_tensor)
                collected_labels.append(label_tensor)

        # After processing all episodes and prefixes, stack the collected data
        print(f"Collect Labels: Actual collected samples = {len(collected_embeddings)}") # DEBUG PRINT
        if collected_embeddings and collected_labels:
            # Check if the number of collected samples matches the expected number
            if len(collected_embeddings) != num_samples:
                 print(f"Warning: Mismatch! Expected {num_samples} samples, but collected {len(collected_embeddings)}.")

            all_comparison_embeddings = torch.stack(collected_embeddings, dim=0) # Shape [num_samples, num_policies, embedding_dim]
            all_labels = torch.stack(collected_labels, dim=0) # Shape [num_samples, num_policies]
            self._collected_data.append((all_comparison_embeddings, all_labels))

    def fit_estimator(self, preloaded_theta=None):
        """
        Fit the estimator with collected data or preloaded theta
        
        Args:
            preloaded_theta: Pre-computed theta parameter to use instead of fitting
        """
        if preloaded_theta is not None:
            # No need to load dummy data anymore, just fit with preloaded theta
            self.estimator.fit(preloaded_theta=preloaded_theta)
            return

        if not self._collected_data:
            print("No data collected for fitting the estimator.")
            return

        # Combine all collected comparison embeddings and labels
        all_embeddings = torch.cat([embeddings for embeddings, _ in self._collected_data], dim=0)
        all_labels = torch.cat([labels for _, labels in self._collected_data], dim=0)

        # Fit the estimator directly with the embeddings and labels
        # No need to load env.emissions or specify sum_dim
        print(f"Fitting MultinomialFeedback estimator with {all_embeddings.shape[0]} samples.")
        self.estimator.fit(comparison_embeddings=all_embeddings, labels=all_labels)

class FeedbackFactory:
    """
    Decides which feedback type to build + design + estimator for numerical or multinomial.
    """
    @staticmethod
    def create(cfg, env, scorer_model, embedder, scorer_model_name=None): # Added scorer_model_name argument
        """
        Creates feedback components.

        Args:
            cfg: Configuration object.
            env: Environment object (LLMGrid).
            scorer_model: The ground truth scorer model instance.
            embedder: The embedder instance.
            scorer_model_name: The name of the scorer model (used for lambda lookup). # Added scorer_model_name to docstring
        """
        # Use embedder's dimension instead of hardcoding
        m = embedder.get_embedding_dim()
        embedding = CustomEmbedding(m, lambda x: x, m)

        # Get the single lambda value
        if not hasattr(cfg.feedback, 'lambda'):
            raise ValueError(f"lambda not found in feedback config '{cfg.feedback.name}'. A single 'lambda' parameter is expected.")
        lambda_val = cfg.feedback['lambda'] # Use dictionary-style access for the key 'lambda'
        print(f"Using single lambda value: {lambda_val} for both design and estimation regularization.")

        # Decide which feedback type
        if cfg.feedback.name == 'numerical':
            # Numerical feedback uses DesignD (A-optimal is similar, D is often preferred)
            design = DesignD(env=env, lambd=lambda_val, dim=1)
            # KernelizedFeatures doesn't use lambda directly in constructor, it's set during fit if needed
            estimator = KernelizedFeatures(embedding, m)
            # If KernelizedFeatures needed lambda_val for estimation, it would be passed during fit or set as attribute
            fb = NumericalFeedback(env, design, estimator)
        else: # Multinomial feedback
            # Parse adaptive_design_frequency
            parsed_adaptive_design_freq = 0
            raw_adf = cfg.feedback.adaptive_design_frequency
            if isinstance(raw_adf, str) and raw_adf.startswith('/'):
                try:
                    divisor = int(raw_adf[1:])
                    if divisor > 0:
                        total_episodes = cfg.experiment.episodes
                        parsed_adaptive_design_freq = total_episodes // divisor
                    else:
                        warnings.warn(f"adaptive_design_frequency divisor must be positive, got {divisor}. Defaulting to 0 (non-adaptive design).")
                except ValueError:
                    warnings.warn(f"Malformed adaptive_design_frequency string '{raw_adf}'. Expected format '/n'. Defaulting to 0 (non-adaptive design).")
                except AttributeError: # Handle missing cfg.experiment.episodes for safety
                    warnings.warn(f"cfg.experiment.episodes not found. Cannot calculate adaptive_design_frequency from '{raw_adf}'. Defaulting to 0 (non-adaptive design).")
            elif isinstance(raw_adf, int):
                parsed_adaptive_design_freq = raw_adf
            else:
                warnings.warn(f"Unexpected type for adaptive_design_frequency: {type(raw_adf)}. Expected int or string like '/n'. Defaulting to 0 (non-adaptive design).")

            # Parse adaptive_estimation_frequency
            parsed_adaptive_estimation_freq = 0
            raw_aef = cfg.feedback.adaptive_estimation_frequency
            if isinstance(raw_aef, str) and raw_aef.startswith('/'):
                try:
                    divisor = int(raw_aef[1:])
                    if divisor > 0:
                        total_episodes = cfg.experiment.episodes
                        parsed_adaptive_estimation_freq = total_episodes // divisor
                    else:
                        warnings.warn(f"adaptive_estimation_frequency divisor must be positive, got {divisor}. Defaulting to 0 (non-adaptive estimation).")
                except ValueError:
                    warnings.warn(f"Malformed adaptive_estimation_frequency string '{raw_aef}'. Expected format '/n'. Defaulting to 0 (non-adaptive estimation).")
                except AttributeError: # Handle missing cfg.experiment.episodes for safety
                    warnings.warn(f"cfg.experiment.episodes not found. Cannot calculate adaptive_estimation_frequency from '{raw_aef}'. Defaulting to 0 (non-adaptive estimation).")
            elif isinstance(raw_aef, int):
                parsed_adaptive_estimation_freq = raw_aef
            else:
                warnings.warn(f"Unexpected type for adaptive_estimation_frequency: {type(raw_aef)}. Expected int or string like '/n'. Defaulting to 0 (non-adaptive estimation).")


            V = None
            if cfg.feedback.pass_V:
                # Calculate V by summing state-specific V_h matrices (used for A-optimal)
                # V_h considers differences only between actions valid for state h
                embedding_dim = env.get_dim()
                V = torch.zeros((embedding_dim, embedding_dim), 
                                dtype=env.emissions.dtype, 
                                device=env.emissions.device)

                for h in range(env.max_episode_length):
                    # Find valid action indices for state h
                    valid_action_indices = [
                        action_idx for action_idx in range(env.actions_num) 
                        if env.is_valid_action(action_idx, h)
                    ]
                    
                    if not valid_action_indices:
                        continue # Skip if no valid actions for this state

                    # Extract corresponding emissions
                    X_h = env.emissions[valid_action_indices, :] # Shape: [n_h x 768]
                    n_h = X_h.shape[0]

                    if n_h <= 1:
                        continue # Need at least 2 actions to compute differences

                    # Compute sum of rows for state h
                    S_h = torch.sum(X_h, dim=0)  # Shape: [768]
                    
                    # Compute X_h.T @ X_h
                    XTX_h = torch.mm(X_h.T, X_h)  # Shape: [768 x 768]
                    
                    # Compute the outer product S_h @ S_h.T
                    S_outer_h = torch.outer(S_h, S_h)  # Shape: [768 x 768]
                    
                    # Compute V_h for state h
                    V_h = 2 * n_h * XTX_h - 2 * S_outer_h  # Shape: [768 x 768]
                    
                    # Add to the total V
                    V += V_h

            # --- Determine Design based on cfg.feedback.objective ---
            design_objective = cfg.feedback.get('objective', 'A').upper() # Default to 'A' if not specified

            if design_objective not in ['A', 'C']:
                warnings.warn(f"Invalid design_objective '{cfg.feedback.get('objective')}'. Defaulting to A-optimal design.")
                design_objective = 'A'

            # --- Select Adaptive or Static Design ---
            if design_objective == 'A':
                if parsed_adaptive_design_freq > 0: # Adaptive A-optimal design
                    design = AdaptiveOrigDesignA(
                        env=env,
                        lambd=lambda_val,
                        dim=1,
                        V=V
                    )
                    print(f"Using Adaptive A-optimal design. Design re-optimization frequency (explorer controlled): {parsed_adaptive_design_freq}.")
                else: # Static A-optimal design
                    design = MultiPolicyOrigDesignA(
                        env=env,
                        lambd=lambda_val,
                        dim=1,
                        V=V
                    )
                    print("Using Static A-optimal design.")
            elif design_objective == 'C':
                if parsed_adaptive_design_freq > 0 and parsed_adaptive_estimation_freq > 0:
                    # Adaptive C-optimal design (both design and C vector are adaptive)
                    design = AdaptiveOrigDesignC(
                        env=env,
                        lambd=lambda_val,
                        dim=1,
                        C=None,  # C will be set by update_estimator
                        adaptive_estimation_frequency=parsed_adaptive_estimation_freq, # This controls C updates
                        update_C_from_estimator=True
                    )
                    print(f"Using Adaptive C-optimal design. C vector updated every {parsed_adaptive_estimation_freq} (estimator) steps. "
                          f"Design re-optimized every {parsed_adaptive_design_freq} (explorer) steps.")
                else:
                    # Error for C-optimal if not fully adaptive
                    raise ValueError("C-optimal design (objective='C') requires both adaptive_design_frequency > 0 AND "
                                     "adaptive_estimation_frequency > 0. "
                                     f"Current settings: adaptive_design_frequency={parsed_adaptive_design_freq}, "
                                     f"adaptive_estimation_frequency={parsed_adaptive_estimation_freq}.")
            # No 'else' needed here as design_objective is guaranteed to be 'A' or 'C' by prior checks.

            likelihood = MultinomialLikelihood()
            regularizer = L2Regularizer(lam=lambda_val) # Use single lambda_val for estimation regularization
            estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)
            fb = MultinomialFeedback(env, design, estimator)

        return fb, design, estimator
