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

    # REINFORCE settings
    parser.add_argument("--num-reinforce-samples", type=int, default=100, help="Inner samples T for REINFORCE gradient estimation")

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
    logger.info(f"REINFORCE samples per iteration (T): {args.num_reinforce_samples}")
    logger.info(f"Policies (K): {args.num_policies}")
    logger.info(f"Prompt prefix: '{args.prompt_prefix}'")
    logger.info("Using CLIP text embeddings (fast, no SD during optimization)")
    logger.info("Each REINFORCE sample: K embeddings (one per policy) → Fisher_t → L_t · ∇log p")

    history = {
        "iterations": [],
        "objectives": [],
        "sample_prompts": [],
    }

    # Get all trainable (LoRA) parameters once
    lora_params = list(policy_manager.get_all_trainable_parameters())
    T = args.num_reinforce_samples
    K = args.num_policies

    for iteration in range(args.num_iterations):
        # Step 1: Generate T samples per policy AT ONCE (batched)
        # all_samples[q][t] = (text, log_prob) for policy q, sample t
        all_samples = policy_manager.generate_samples(
            num_samples_per_policy=T,
            prompt_prefix=args.prompt_prefix,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Extract texts and log_probs: all_texts[q][t], all_log_probs[q][t]
        all_texts = [[text for text, _ in samples_q] for samples_q in all_samples]
        all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

        # Step 2: Compute CLIP embeddings for all K×T prompts (batched)
        # all_embeddings[q] = list of T embeddings for policy q
        all_embeddings = []
        for q in range(K):
            embeddings_q = embed_prompts_with_clip_text(all_texts[q], embedder)
            all_embeddings.append(embeddings_q)

        # Step 3: Compute T Fisher matrices and accumulate weighted gradients
        accumulated_grads = [torch.zeros_like(p) for p in lora_params]
        fisher_sum = 0.0

        for t in range(T):
            # Fisher_t from K embeddings at time t (one per policy)
            embeddings_t = [[all_embeddings[q][t]] for q in range(K)]
            with torch.no_grad():
                L_t = fisher_objective.compute_objective(embeddings_t)
            fisher_sum += L_t.item()

            # log_prob_t = sum of K log_probs at time t
            log_prob_t = sum(all_log_probs[q][t] for q in range(K))

            # Gradient of log_prob_t w.r.t. LoRA params
            grads_t = torch.autograd.grad(log_prob_t, lora_params, retain_graph=True)

            # Accumulate: L_t · ∇log p_t
            for i, g in enumerate(grads_t):
                accumulated_grads[i] += L_t * g

        # Average the accumulated gradients
        avg_fisher = fisher_sum / T
        for i in range(len(accumulated_grads)):
            accumulated_grads[i] /= T

        # Apply REINFORCE update: θ += lr * E[L · ∇log p]
        # Optimizer does descent, so set grad = -accumulated
        optimizer.zero_grad()
        for param, acc_g in zip(lora_params, accumulated_grads):
            param.grad = -acc_g  # Negative for ascent
        optimizer.step()

        # Record history
        history["iterations"].append(iteration)
        history["objectives"].append(avg_fisher)
        history["sample_prompts"].append(all_texts[0][0])

        # Logging
        if iteration % args.log_interval == 0 or iteration == args.num_iterations - 1:
            logger.info(
                f"Iteration {iteration:3d}/{args.num_iterations}: "
                f"avg Fisher = {avg_fisher:.4f}"
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
