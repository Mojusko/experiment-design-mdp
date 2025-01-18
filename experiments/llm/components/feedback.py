import torch
import torch.nn.functional as F
import numpy as np

from doexpy.functionals.doe_static_functionals import (
    DesignA, MultiPolicyOrigDesignD, MultiPolicyAggDesignD
)
from doexpy.feedback.feedback_base import EmptyFeedback
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.regression.kernelized_features import KernelizedFeatures
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
from stpy.probability.multinomial_likelihood import MultinomialLikelihood
from stpy.regularization.regularizer import L2Regularizer

class BaseFeedback:
    """
    Base class that handles data collection from visits
    and calls estimator.fit().
    """
    def __init__(self, env, design, estimator):
        self.env = env
        self.design = design
        self.estimator = estimator
        self.feedback = EmptyFeedback(env, design)  # If you need custom feedback logic

    def collect_data(self, cfg, visits, estimator, theta_star):
        raise NotImplementedError

class NumericalFeedback(BaseFeedback):
    """Collect data for numerical feedback, then fit."""
    def collect_data(self, cfg, visits, estimator, theta_star):
        horizon = cfg.horizon
        prefix_range = range(1, horizon+1) if cfg.dense_feedback else range(horizon, horizon+1)

        x_list, y_list = [], []
        for ep in range(cfg.experiment.episodes):
            actions = visits[ep][1]  # (states, actions)
            for prefix_len in prefix_range:
                truncated = actions[:prefix_len]
                yy, xx = theta_star(truncated)
                x_list.append(xx.detach().cpu())
                y_list.append(yy.detach().cpu())
        x_torch = torch.vstack(x_list)
        y_torch = torch.vstack(y_list)

        estimator.load_data((x_torch, y_torch))
        estimator.fit()

class MultinomialFeedback(BaseFeedback):
    """Collect data for multinomial feedback, then fit."""
    def collect_data(self, cfg, visits, estimator, theta_star):
        horizon = cfg.horizon
        num_policies = cfg.feedback.num_policies

        prefix_range = range(1, horizon+1) if cfg.dense_feedback else range(horizon, horizon+1)
        num_samples = cfg.experiment.episodes * (horizon if cfg.dense_feedback else 1)

        trajectory_indices = torch.zeros((num_samples, horizon, num_policies), dtype=torch.long)
        labels = torch.zeros((num_samples, num_policies))
        sample_idx = 0
        for ep in range(cfg.experiment.episodes):
            policy_actions = [visits[p][ep][1] for p in range(num_policies)]
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

        estimator.load_data((self.env.emissions.detach().cpu(), torch.zeros(len(self.env.emissions))))
        estimator.fit(trajectory_indices, labels, sum_dim=1)


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
            design = DesignA(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            estimator = KernelizedFeatures(embedding, m)
            fb = NumericalFeedback(env, design, estimator)
        else:
            #design = MultiPolicyOrigDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            design = MultiPolicyAggDesignD(env=env, lambd=cfg.feedback.lambda_reg, dim=1)
            likelihood = MultinomialLikelihood()
            regularizer = L2Regularizer(lam=cfg.feedback.lambda_reg)
            estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)
            fb = MultinomialFeedback(env, design, estimator)

        return fb, design, estimator
