"""
Monte Carlo Fisher Information estimation for REINFORCE-based experimental design.

This module computes Fisher Information from sampled embeddings and provides
the REINFORCE loss for policy gradient optimization.

The objective is D-optimal: L = log det(Î)

REINFORCE gradient: ∇E[L] ≈ L · (1/N) Σ_j ∇log p(τ_j)
"""

import torch
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MonteCarloFisherObjective:
    """
    Computes Fisher Information and REINFORCE loss from sampled embeddings.

    The Fisher Information is estimated via Monte Carlo:
        Î = (1/K) Σ_q Σ̂_q - μ̄ μ̄ᵀ + λI

    where:
        Σ̂_q = (1/N) Σ_j φ_q^{(j)} (φ_q^{(j)})ᵀ   (second moment for policy q)
        μ̄ = (1/K) Σ_q [(1/N) Σ_j φ_q^{(j)}]      (grand mean)

    The objective is D-optimal: L = log det(Î)

    REINFORCE loss: -L · (1/KN) Σ_{q,j} log p(τ_q^{(j)})
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        embedding_dim: int = 768,
        device: str = "cuda",
    ):
        """
        Initialize the Fisher objective.

        Args:
            lambda_reg: Regularization parameter (λ in Î + λI)
            embedding_dim: Dimension of CLIP embeddings
            device: Device for computations
        """
        self.lambda_reg = lambda_reg
        self.embedding_dim = embedding_dim
        self.device = device

    def compute_fisher_info(
        self,
        all_embeddings: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute Fisher Information matrix from sampled embeddings.

        Args:
            all_embeddings: List of K lists, each with N embeddings.
                           all_embeddings[q][j] = φ(τ_q^{(j)}) ∈ ℝ^d

        Returns:
            Fisher Information matrix Î ∈ ℝ^{d×d}
        """
        K = len(all_embeddings)
        d = all_embeddings[0][0].shape[0]

        # Compute per-policy statistics
        second_moments = []
        means = []

        for q in range(K):
            # Stack embeddings for policy q: (N, d)
            embeddings_q = torch.stack(all_embeddings[q])
            N = embeddings_q.shape[0]

            # Mean: (d,)
            mean_q = embeddings_q.mean(dim=0)
            means.append(mean_q)

            # Second moment: (d, d)
            # Σ̂_q = (1/N) Σ_j φ φᵀ = (1/N) Φᵀ Φ where Φ is (N, d)
            second_q = (embeddings_q.T @ embeddings_q) / N
            second_moments.append(second_q)

        # Average second moment across policies
        avg_second = sum(second_moments) / K

        # Grand mean
        mu_bar = sum(means) / K

        # Fisher Information
        # Î = avg_second - μ̄ μ̄ᵀ + λI
        I_hat = avg_second - torch.outer(mu_bar, mu_bar)
        I_hat = I_hat + self.lambda_reg * torch.eye(d, device=self.device, dtype=I_hat.dtype)

        return I_hat

    def compute_objective(
        self,
        all_embeddings: List[List[torch.Tensor]],
        design: str = "A",
    ) -> torch.Tensor:
        """
        Compute optimal design objective.

        Args:
            all_embeddings: Sampled embeddings from all policies.
            design: "D" for log det(Î), "A" for -tr(Î⁻¹)

        Returns:
            Scalar objective value (to be maximized)
        """
        I_hat = self.compute_fisher_info(all_embeddings)

        if design == "D":
            # D-optimal: log det(Î)
            sign, logdet = torch.linalg.slogdet(I_hat)
            if sign <= 0:
                logger.warning(f"Fisher Information matrix is not positive definite (sign={sign})")
                return torch.tensor(float('-inf'), device=self.device)
            return logdet
        elif design == "A":
            # A-optimal: -tr(Î⁻¹)
            I_inv = torch.linalg.inv(I_hat)
            return -torch.trace(I_inv)
        else:
            raise ValueError(f"Unknown design: {design}")

    def compute_reinforce_loss(
        self,
        all_embeddings: List[List[torch.Tensor]],
        all_log_probs: List[List[torch.Tensor]],
        baseline: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute REINFORCE loss for policy gradient update.

        Loss = -L · (1/KN) Σ_{q,j} log p(τ_q^{(j)})

        where L = log det(Î) is the D-optimal objective.

        Args:
            all_embeddings: Sampled embeddings from all policies.
            all_log_probs: Log-probabilities from all policies.
                          all_log_probs[q][j] = log p_{θ_q}(τ_q^{(j)})
            baseline: Optional baseline to subtract from L for variance reduction
                     (e.g., moving average of past objectives).

        Returns:
            Tuple of (loss, objective):
                - loss: Scalar loss tensor (with gradients through log_probs)
                - objective: The D-optimal objective L (for logging)
        """
        # Compute objective L = log det(Î)
        with torch.no_grad():
            objective = self.compute_objective(all_embeddings)

        # Sum all log probabilities
        total_log_prob = torch.tensor(0.0, device=self.device)
        total_samples = 0

        K = len(all_embeddings)
        for q in range(K):
            N = len(all_embeddings[q])
            for j in range(N):
                total_log_prob = total_log_prob + all_log_probs[q][j]
                total_samples += 1

        # Average log probability
        avg_log_prob = total_log_prob / total_samples

        # REINFORCE loss: -L · avg_log_prob
        # (We maximize L by minimizing -L · log_prob)
        L_centered = objective - baseline
        loss = -L_centered.detach() * avg_log_prob

        return loss, objective


def embed_prompts_with_clip_text(
    prompts: List[str],
    embedder,
) -> List[torch.Tensor]:
    """
    Embed prompts using CLIP text encoder.

    This is fast (no image generation needed) and used during REINFORCE optimization.
    CLIP text and image embeddings are aligned, so text embeddings are a good proxy
    for optimizing Fisher Information.

    Args:
        prompts: List of text prompts
        embedder: CLIPEmbedder instance (must have embed_text method)

    Returns:
        List of embedding tensors, each of shape (d,)
    """
    embeddings = []

    for prompt in prompts:
        # Use CLIP text encoder directly - no SD needed!
        embedding = embedder.embed_text(prompt)  # (1, d)
        embeddings.append(embedding.squeeze(0))  # (d,)

    return embeddings
