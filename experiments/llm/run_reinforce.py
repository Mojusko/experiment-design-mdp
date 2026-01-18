#!/usr/bin/env python3
"""
Standalone REINFORCE experiment for vocabulary-free experimental design.

This script demonstrates the full REINFORCE approach:
1. K GPT-2 policies with LoRA adapters
2. Sample prompts from each policy
3. Compute CLIP text embeddings (fast, no SD)
4. Estimate Fisher Information (Monte Carlo)
5. Update policies via REINFORCE

After optimization, prompts can be used for SD image generation and feedback collection.

Usage:
    python run_reinforce.py [options]
"""

import os
import sys
import argparse
import logging
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.gpt2_generative_policy import MultiPolicyManager
from components.monte_carlo_fisher import MonteCarloFisherObjective, embed_prompts_with_clip_text
from components.embedder import CLIPEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE for vocabulary-free experimental design")

    # GPT-2 settings
    parser.add_argument("--model-name", type=str, default="gpt2", help="GPT-2 model name")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")

    # Optimization settings
    parser.add_argument("--num-policies", type=int, default=4, help="Number of policies (K)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Optimization iterations")
    parser.add_argument("--samples-per-policy", type=int, default=16, help="Samples per policy (N)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Fisher regularization")

    # Generation settings
    parser.add_argument("--prompt-prefix", type=str, default="", help="Prompt prefix (maps to base_prompt in config)")
    parser.add_argument("--max-new-tokens", type=int, default=15, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    # Output settings
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.output_dir = f"results/reinforce-{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("REINFORCE Vocabulary-Free Experimental Design")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Initialize components
    logger.info("\n--- Initializing Components ---")

    # 1. Create K policies
    logger.info(f"Creating {args.num_policies} GPT-2 policies with LoRA (rank={args.lora_rank})")
    policy_manager = MultiPolicyManager(
        num_policies=args.num_policies,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )

    # 2. Create optimizer
    optimizer = torch.optim.AdamW(
        policy_manager.get_all_trainable_parameters(),
        lr=args.learning_rate,
    )

    # 3. Create Fisher objective
    fisher_objective = MonteCarloFisherObjective(
        lambda_reg=args.lambda_reg,
        device=args.device,
    )

    # 4. Create CLIP embedder (text encoder only - no SD needed for optimization)
    logger.info("Loading CLIP text embedder...")
    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        normalize=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )

    # Sanity check: verify all policies produce similar distributions before optimization
    logger.info("\n--- Sanity Check: Initial Policy Similarity ---")
    with torch.no_grad():
        test_input = policy_manager.policies[0].tokenizer(
            args.prompt_prefix if args.prompt_prefix else "A photo of",
            return_tensors="pt"
        ).to(args.device)

        all_logits = []
        for q, policy in enumerate(policy_manager.policies):
            outputs = policy.model(**test_input)
            logits = outputs.logits[0, -1, :]  # Last token logits
            all_logits.append(logits)

        # Compare distributions via cosine similarity
        for i in range(len(all_logits)):
            for j in range(i + 1, len(all_logits)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    all_logits[i].unsqueeze(0),
                    all_logits[j].unsqueeze(0)
                ).item()
                logger.info(f"  Policy {i} vs Policy {j}: cosine similarity = {cos_sim:.6f}")

        # Check they're similar (should be ~1.0 at initialization)
        avg_sim = sum(
            torch.nn.functional.cosine_similarity(all_logits[i].unsqueeze(0), all_logits[j].unsqueeze(0)).item()
            for i in range(len(all_logits)) for j in range(i + 1, len(all_logits))
        ) / max(1, len(all_logits) * (len(all_logits) - 1) // 2)
        logger.info(f"  Average similarity: {avg_sim:.6f} (should be ~1.0 at init)")

    # Training loop
    logger.info("\n--- Starting Optimization ---")
    logger.info(f"Iterations: {args.num_iterations}")
    logger.info(f"Samples per policy: {args.samples_per_policy}")
    logger.info(f"Prompt prefix: '{args.prompt_prefix}'")
    logger.info("Using CLIP text embeddings (fast, no SD during optimization)")

    history = {
        "iterations": [],
        "objectives": [],
        "losses": [],
        "sample_prompts": [],
    }

    for iteration in range(args.num_iterations):
        # Step 1: Sample prompts from each policy
        all_samples = policy_manager.generate_samples(
            num_samples_per_policy=args.samples_per_policy,
            prompt_prefix=args.prompt_prefix,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Extract texts and log_probs
        all_texts = [[text for text, _ in samples_q] for samples_q in all_samples]
        all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

        # Step 2: Compute CLIP text embeddings (fast!)
        all_embeddings = []
        for q in range(args.num_policies):
            embeddings_q = embed_prompts_with_clip_text(all_texts[q], embedder)
            all_embeddings.append(embeddings_q)

        # Step 3: Compute Fisher objective and avg_log_prob
        loss, objective, avg_log_prob = fisher_objective.compute_reinforce_loss(
            all_embeddings,
            all_log_probs,
        )

        # Step 4: Explicit REINFORCE gradient
        # Get all trainable (LoRA) parameters
        lora_params = policy_manager.get_all_trainable_parameters()

        # Compute sum of log probs (need gradient through this)
        total_log_prob = sum(lp for lps in all_log_probs for lp in lps)

        # Get gradient of log_prob w.r.t. LoRA params
        grads = torch.autograd.grad(total_log_prob, lora_params)

        # REINFORCE: gradient = L * grad_log_prob (for maximizing L)
        # We want to maximize Fisher, so gradient ascent: θ += lr * L * ∇log_p
        # Optimizer does gradient descent: θ -= lr * param.grad
        # So we set param.grad = -L * ∇log_p
        optimizer.zero_grad()
        L = objective.detach()
        for param, g in zip(lora_params, grads):
            param.grad = -L * g
        optimizer.step()

        # Record history
        history["iterations"].append(iteration)
        history["objectives"].append(objective.item())
        history["losses"].append(loss.item())
        history["sample_prompts"].append(all_texts[0][0])

        # Logging
        if iteration % args.log_interval == 0 or iteration == args.num_iterations - 1:
            logger.info(
                f"Iteration {iteration:3d}/{args.num_iterations}: "
                f"Fisher = {objective.item():.4f}, "
                f"avg_logp = {avg_log_prob.item():.2f}, "
                f"Loss = {loss.item():.4f}"
            )

    # Save results
    logger.info("\n--- Saving Results ---")

    # Save policies
    policy_path = os.path.join(args.output_dir, "policies.pt")
    state = {
        "policies": [p.model.state_dict() for p in policy_manager.policies],
        "args": vars(args),
        "history": history,
    }
    torch.save(state, policy_path)
    logger.info(f"Saved policies to {policy_path}")

    # Generate final prompts for evaluation
    logger.info("\n--- Generating Final Prompts (for later SD generation) ---")
    final_prompts = []
    for q, policy in enumerate(policy_manager.policies):
        prompts_q = []
        for _ in range(10):
            with torch.no_grad():
                text, _ = policy.generate_with_log_prob(
                    prompt_prefix=args.prompt_prefix,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                )
            prompts_q.append(text)
        final_prompts.append(prompts_q)
        logger.info(f"Policy {q} samples:")
        for i, p in enumerate(prompts_q[:3]):
            logger.info(f"  {i}: {p}")

    # Save final prompts
    prompts_path = os.path.join(args.output_dir, "final_prompts.txt")
    with open(prompts_path, "w") as f:
        for q, prompts_q in enumerate(final_prompts):
            f.write(f"=== Policy {q} ===\n")
            for p in prompts_q:
                f.write(f"{p}\n")
            f.write("\n")
    logger.info(f"Saved final prompts to {prompts_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Optimization Complete!")
    logger.info(f"Initial Fisher Info: {history['objectives'][0]:.4f}")
    logger.info(f"Final Fisher Info: {history['objectives'][-1]:.4f}")
    improvement = history['objectives'][-1] - history['objectives'][0]
    logger.info(f"Improvement: {improvement:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Use final_prompts.txt to generate images with SD")
    logger.info("  2. Collect user feedback on the generated images")
    logger.info("  3. Estimate theta from feedback")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
