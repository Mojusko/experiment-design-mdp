import torch
import torch.nn.functional as F
import numpy as np
from doexpy.env.llm import create_prompt, create_prompt_from_tokens

from doexpy.functionals.doe_static_functionals import (
    DesignA, DesignD, MultiPolicyOrigDesignA, MultiPolicyOrigDesignD, MultiPolicyOrigDesignC
)
from doexpy.functionals.doe_adaptive_functionals import (
    AdaptiveOrigDesignD, AdaptiveOrigDesignA, AdaptiveOrigDesignC, AdaptiveOrigDesignANovel
)
from doexpy.feedback.feedback_base import EmptyFeedback
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.regression.kernelized_features import KernelizedFeatures
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
from stpy.probability.multinomial_likelihood import MultinomialLikelihood
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
        horizon = cfg.horizon
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
        horizon = cfg.horizon
        num_policies = cfg.feedback.num_policies
        num_episodes = len(new_visits[0])
        prefix_range = range(1, horizon+1) if cfg.dense_feedback else range(horizon, horizon+1)
        num_samples = num_episodes * (horizon if cfg.dense_feedback else 1)

        prob_products = []

        trajectory_indices = torch.zeros((num_samples, horizon, num_policies), dtype=torch.long)
        labels = torch.zeros((num_samples, num_policies))
        sample_idx = 0

        for ep in range(num_episodes):
            policy_actions = [visits[ep][1] for visits in new_visits]
            for prefix_len in prefix_range:
                trunc_actions = [
                    act[:prefix_len] + [0]*(horizon-prefix_len) for act in policy_actions
                ]
                vals = torch.tensor([theta_star(ta)[0] for ta in trunc_actions])
                probs = F.softmax(vals.detach(), dim=0)
                label_idx = torch.multinomial(probs, 1)

                trajectory_indices[sample_idx, :, :] = torch.tensor(trunc_actions).T
                labels[sample_idx, label_idx] = 1
                sample_idx += 1

                if num_policies == 2:
                    prob_products.append((probs[0] * probs[1]).item())

        self._collected_data.append((trajectory_indices, labels))

    def fit_estimator(self, preloaded_theta=None):
        """
        Fit the estimator with collected data or preloaded theta
        
        Args:
            preloaded_theta: Pre-computed theta parameter to use instead of fitting
        """
        if preloaded_theta is not None:
            # Load emissions first to ensure the estimator has the right dimensions
            self.estimator.load_data((self.env.emissions.detach().cpu(), torch.zeros(len(self.env.emissions))))
            # Then use preloaded theta directly
            self.estimator.fit(preloaded_theta=preloaded_theta)
            return
            
        if not self._collected_data:
            return
            
        # Combine all trajectory indices and labels
        all_indices = torch.cat([indices for indices, _ in self._collected_data], dim=0)
        all_labels = torch.cat([labels for _, labels in self._collected_data], dim=0)

        # Load emissions and fit with combined data
        self.estimator.load_data((self.env.emissions.detach().cpu(), torch.zeros(len(self.env.emissions))))
        self.estimator.fit(all_indices, all_labels, sum_dim=1)

class FeedbackFactory:
    """
    Decides which feedback type to build + design + estimator for numerical or multinomial.
    """
    @staticmethod
    def create(cfg, env, scorer_model):
        """
        Creates feedback components.

        Args:
            cfg: Configuration object.
            env: Environment object (LLMGrid).
            scorer_model: The ground truth scorer model instance.
        """
        m = 768
        embedding = CustomEmbedding(m, lambda x: x, m)

        # Determine lambda_reg based on the configuration strategy
        if cfg.feedback.name == 'multinomial' and cfg.feedback.use_model_specific_lambda:
            # Use model-specific lambda from the dictionary
            scorer_model_name = cfg.experiment.scorer_model
            model_params = cfg.feedback.model_specific_params.get(scorer_model_name, cfg.feedback.lambda_reg)
            lambda_reg = model_params.lambda_reg
            print(f"Using model-specific lambda_reg = {lambda_reg} for scorer_model = {scorer_model_name}")
        else:
            # Use the general lambda_reg (either for numerical or if flag is false for multinomial)
            # Ensure lambda_reg exists in the feedback config
            if not hasattr(cfg.feedback, 'lambda_reg'):
                 raise ValueError(f"lambda_reg not found in feedback config '{cfg.feedback.name}' and use_model_specific_lambda is false or feedback is not multinomial.")
            lambda_reg = cfg.feedback.lambda_reg
            print(f"Using general lambda_reg = {lambda_reg} (numerical feedback or use_model_specific_lambda=false)")


        # Decide which feedback type
        if cfg.feedback.name == 'numerical':
            #design = DesignA(env=env, lambd=lambda_reg, dim=1) # Use determined lambda_reg
            design = DesignD(env=env, lambd=lambda_reg, dim=1) # Use determined lambda_reg
            estimator = KernelizedFeatures(embedding, m)
            fb = NumericalFeedback(env, design, estimator)
        else: # Multinomial feedback

            #design = MultiPolicyOrigDesignD(env=env, lambd=lambda_reg, dim=1) # Use determined lambda_reg
            V = None
            if cfg.feedback.pass_V:
                # Calculate V by summing state-specific V_h matrices
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
            
            # Determine the initial C vector. Currently using GT scorer weights directly.
            initial_C = scorer_model.weight.data # Use passed scorer_model
            # initial_C = env.get_prior_vector() # Uncomment to use the generated prior vector instead

            if cfg.feedback.adaptive_design_frequency > 0:
                # Pass initial_C and estimation frequency to the adaptive design constructor
                #design = AdaptiveOrigDesignC(
                #    env=env,
                #    lambd=lambda_reg, # Use determined lambda_reg
                #    dim=1,
                #    C=initial_C,
                #    adaptive_estimation_frequency=cfg.feedback.adaptive_estimation_frequency # Pass frequency
                #)
                #design = AdaptiveOrigDesignA(env=env, lambd=lambda_reg, dim=1, V=V) # Use determined lambda_reg
                design = AdaptiveOrigDesignA(env=env, lambd=lambda_reg, dim=1, V=V) # Use determined lambda_reg
                #design = AdaptiveOrigDesignD(env=env, lambd=lambda_reg, dim=1) # Use determined lambda_reg
            else:
                # Static designs
                design = MultiPolicyOrigDesignA(env=env, lambd=lambda_reg, dim=1,V=V) # Use determined lambda_reg
                #design = MultiPolicyOrigDesignD(env=env, lambd=lambda_reg, dim=1) # Use determined lambda_reg
                # The MultiPolicyOrigDesignC constructor will raise ValueError if initial_C is None.
                #design = MultiPolicyOrigDesignC(env=env, lambd=lambda_reg, dim=1, C=initial_C) # Use determined lambda_reg

            likelihood = MultinomialLikelihood()
            regularizer = L2Regularizer(lam=lambda_reg) # Use determined lambda_reg
            estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)
            fb = MultinomialFeedback(env, design, estimator)

        return fb, design, estimator
