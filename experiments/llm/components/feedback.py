import torch
import torch.nn.functional as F
import numpy as np

from doexpy.functionals.doe_static_functionals import (
    DesignA, DesignD, MultiPolicyOrigDesignA, MultiPolicyOrigDesignD
)
from doexpy.functionals.doe_adaptive_functionals import (
    AdaptiveOrigDesignD, StochasticAdaptiveOrigDesignA, StochasticAdaptiveOrigDesignD, 
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
        self._metrics = {}
        self._collected_data = []
    
    @property
    def metrics(self):
        return self._metrics

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

    def fit_estimator(self):
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

        self._metrics = {
            'mean_prob_product': np.mean(prob_products),
            'std_prob_product': np.std(prob_products)
        }

        self._collected_data.append((trajectory_indices, labels))

    def fit_estimator(self):
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
    def create(cfg, env):
        m = 768
        embedding = CustomEmbedding(m, lambda x: x, m)

        # Decide which feedback type
        if cfg.feedback.name == 'numerical':
            #design = DesignA(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            design = DesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            estimator = KernelizedFeatures(embedding, m)
            fb = NumericalFeedback(env, design, estimator)
        else:

            #design = MultiPolicyOrigDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            if cfg.feedback.pass_V:
                # Create matrix A of all pairwise differences
                n = env.emissions.shape[0]
                rows = []
                for i in range(n):
                    for j in range(n):
                        diff = env.emissions[i] - env.emissions[j]
                        rows.append(diff)
                A = torch.stack(rows)  # Shape: [n*(n-1)/2 x 768]
                
                # Compute V using the difference matrix A
                V = torch.mm(A.T, A)  # Shape: [768 x 768]
            else:
                V=None
            if cfg.feedback.adaptive_design_frequency > 0:
                design = StochasticAdaptiveOrigDesignA(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
                #design = StochasticAdaptiveOrigDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
                #design = AdaptiveOrigDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1) 

                #design = AdaptiveOrigDesignC(env=env, lambd=cfg.feedback.lambda_reg, dim=1) 
            else:
                design = MultiPolicyOrigDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1,V=V)

            likelihood = MultinomialLikelihood()
            regularizer = L2Regularizer(lam=cfg.feedback.lambda_reg)
            estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)
            fb = MultinomialFeedback(env, design, estimator)

        return fb, design, estimator
