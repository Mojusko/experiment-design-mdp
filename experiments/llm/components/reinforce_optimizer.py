"""
REINFORCE optimizer for vocabulary-free experimental design.

This module implements the optimization loop that:
1. Samples prompts from K GPT-2 policies
2. Computes CLIP text embeddings (fast, no image generation)
3. Estimates Fisher Information (Monte Carlo)
4. Updates policies via REINFORCE

After optimization, the trained policies are used to generate prompts for
exploration. Only then are images generated (with SD) for feedback collection.

This replaces the Frank-Wolfe optimization used in the fixed-vocabulary setting.
"""

import torch
import torch.optim as optim
from typing import List, Tuple, Optional, Callable
import logging
from tqdm import tqdm

from components.gpt2_generative_policy import MultiPolicyManager, GPT2GenerativePolicy
from components.monte_carlo_fisher import MonteCarloFisherObjective, embed_prompts_with_clip_text
from components.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class REINFORCEOptimizer:
    """
    Optimizes K GPT-2 policies to maximize Fisher Information using REINFORCE.

    This is the vocabulary-free replacement for Frank-Wolfe optimization.
    Instead of optimizing over a fixed vocabulary, we optimize GPT-2 to generate
    informative prompts directly.

    Key insight: During optimization, we use CLIP text embeddings (fast).
    SD image generation only happens after optimization, during exploration.
    """

    def __init__(
        self,
        num_policies: int = 4,
        model_name: str = "gpt2",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lambda_reg: float = 1.0,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        embedder: Optional[BaseEmbedder] = None,
    ):
        """
        Initialize the REINFORCE optimizer.

        Args:
            num_policies: Number of policies (K)
            model_name: GPT-2 model name
            lora_rank: LoRA rank for each policy
            lora_alpha: LoRA alpha for each policy
            lambda_reg: Regularization for Fisher Information (λ)
            learning_rate: Learning rate for LoRA parameters
            device: Device for computation
            embedder: CLIPEmbedder instance (required for text embedding)
        """
        self.device = device
        self.num_policies = num_policies
        self.lambda_reg = lambda_reg

        # Initialize K policies
        logger.info(f"Initializing {num_policies} GPT-2 policies with LoRA")
        self.policy_manager = MultiPolicyManager(
            num_policies=num_policies,
            model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=device,
        )

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy_manager.get_all_trainable_parameters(),
            lr=learning_rate,
        )

        # Fisher objective
        self.fisher_objective = MonteCarloFisherObjective(
            lambda_reg=lambda_reg,
            device=device,
        )

        # CLIP embedder for text embeddings (no SD needed during optimization)
        self._embedder = embedder

    @property
    def embedder(self) -> BaseEmbedder:
        """Get embedder (must be provided)."""
        if self._embedder is None:
            raise ValueError("Embedder must be provided for CLIP text encoding")
        return self._embedder

    def optimize(
        self,
        num_iterations: int = 100,
        samples_per_policy: int = 16,
        prompt_prefix: str = "",
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        baseline: str = "per_policy",
        log_interval: int = 10,
        callback: Optional[Callable[[int, float, List[str]], None]] = None,
    ) -> dict:
        """
        Run the REINFORCE optimization loop.

        Args:
            num_iterations: Number of optimization steps
            samples_per_policy: Number of samples (N) per policy per iteration
            prompt_prefix: Starting text for generation (e.g., "A photo of")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            baseline: Baseline method for variance reduction
            log_interval: How often to log progress
            callback: Optional callback(iteration, objective, sample_prompts)

        Returns:
            Dictionary with optimization history
        """
        history = {
            "iterations": [],
            "objectives": [],
            "sample_prompts": [],
        }

        logger.info(f"Starting REINFORCE optimization for {num_iterations} iterations")
        logger.info(f"  Policies: {self.num_policies}, Samples/policy: {samples_per_policy}")
        logger.info(f"  Prompt prefix: '{prompt_prefix}', Max tokens: {max_new_tokens}")

        for iteration in tqdm(range(num_iterations), desc="REINFORCE"):
            # Step 1: Sample prompts from each policy
            all_samples = self.policy_manager.generate_samples(
                num_samples_per_policy=samples_per_policy,
                prompt_prefix=prompt_prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Extract texts and log_probs
            all_texts = [[text for text, _ in samples_q] for samples_q in all_samples]
            all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

            # Step 2: Compute CLIP text embeddings (fast, no SD needed)
            all_embeddings = []
            for q in range(self.num_policies):
                embeddings_q = embed_prompts_with_clip_text(
                    all_texts[q],
                    self.embedder,
                )
                all_embeddings.append(embeddings_q)

            # Step 3: Compute objective (for logging)
            with torch.no_grad():
                objective = self.fisher_objective.compute_objective(all_embeddings)

            # Step 4: Compute REINFORCE loss
            loss = self.fisher_objective.compute_reinforce_loss(
                all_embeddings,
                all_log_probs,
                baseline=baseline,
            )

            # Step 5: Update policies
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            history["iterations"].append(iteration)
            history["objectives"].append(objective.item())
            history["sample_prompts"].append(all_texts[0][0])  # First sample from first policy

            if iteration % log_interval == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Fisher Info = {objective.item():.4f}, "
                    f"Loss = {loss.item():.4f}"
                )
                # Log a sample prompt from each policy
                for q in range(self.num_policies):
                    logger.info(f"  Policy {q}: '{all_texts[q][0][:60]}...'")

            if callback is not None:
                callback(iteration, objective.item(), [t[0] for t in all_texts])

        return history

    def generate_evaluation_prompts(
        self,
        num_prompts_per_policy: int = 10,
        prompt_prefix: str = "",
        max_new_tokens: int = 20,
    ) -> List[List[str]]:
        """
        Generate prompts from trained policies for evaluation.

        Args:
            num_prompts_per_policy: Number of prompts to generate per policy
            prompt_prefix: Starting text for generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of K lists of generated prompts.
        """
        all_prompts = []

        for q, policy in enumerate(self.policy_manager.policies):
            prompts_q = []
            for _ in range(num_prompts_per_policy):
                with torch.no_grad():
                    text, _ = policy.generate_with_log_prob(
                        prompt_prefix=prompt_prefix,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,  # Lower temperature for more coherent outputs
                    )
                prompts_q.append(text)
            all_prompts.append(prompts_q)

        return all_prompts

    def save_policies(self, path: str):
        """Save policy LoRA weights."""
        state = {
            "num_policies": self.num_policies,
            "policies": [
                policy.model.state_dict()
                for policy in self.policy_manager.policies
            ],
        }
        torch.save(state, path)
        logger.info(f"Saved policies to {path}")

    def load_policies(self, path: str):
        """Load policy LoRA weights."""
        state = torch.load(path, map_location=self.device)
        for q, policy in enumerate(self.policy_manager.policies):
            policy.model.load_state_dict(state["policies"][q])
        logger.info(f"Loaded policies from {path}")
