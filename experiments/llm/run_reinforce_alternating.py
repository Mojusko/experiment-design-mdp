#!/usr/bin/env python3
"""
REINFORCE with ALTERNATING policy updates (Frank-Wolfe style).

Key difference from run_reinforce.py:
- Each iteration updates only ONE policy (cycling through K policies)
- After each update, resample to get fresh Fisher values
- This creates coupling between policies like Frank-Wolfe

Usage:
    python run_reinforce_alternating.py [options]
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
    parser = argparse.ArgumentParser(description="REINFORCE with alternating updates")

    # GPT-2 settings
    parser.add_argument("--model-name", type=str, default="gpt2", help="GPT-2 model name")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")

    # Optimization settings
    parser.add_argument("--num-policies", type=int, default=4, help="Number of policies (K)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Fisher regularization")

    # Generation settings
    parser.add_argument("--prompt-prefix", type=str, default="", help="Prompt prefix")
    parser.add_argument("--max-new-tokens", type=int, default=15, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    # REINFORCE settings
    parser.add_argument("--num-reinforce-samples", type=int, default=50, help="Inner samples T")

    # Output settings
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log-interval", type=int, default=5, help="Logging interval")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.output_dir = f"results/reinforce-alternating-{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("REINFORCE with ALTERNATING Updates (Frank-Wolfe style)")
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

    K = args.num_policies
    T = args.num_reinforce_samples

    # 2. Create SEPARATE optimizer for each policy
    optimizers = []
    for q in range(K):
        opt = torch.optim.AdamW(
            policy_manager.policies[q].get_trainable_parameters(),
            lr=args.learning_rate,
        )
        optimizers.append(opt)
    logger.info(f"Created {K} separate optimizers (one per policy)")

    # 3. Create Fisher objective
    fisher_objective = MonteCarloFisherObjective(
        lambda_reg=args.lambda_reg,
        device=args.device,
    )

    # 4. Create CLIP embedder
    logger.info("Loading CLIP text embedder...")
    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        normalize=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )

    # Training loop
    logger.info("\n--- Starting Alternating Optimization ---")
    logger.info(f"Total iterations: {args.num_iterations}")
    logger.info(f"REINFORCE samples per iteration (T): {T}")
    logger.info(f"Policies (K): {K}")
    logger.info(f"Update schedule: policy (iter % K) updated each iteration")
    logger.info("After each update, resample → fresh Fisher → coupled gradients!")

    history = {
        "iterations": [],
        "objectives": [],
        "updated_policy": [],
    }

    for iteration in range(args.num_iterations):
        # Which policy to update this iteration
        policy_to_update = iteration % K

        # Step 1: Generate T samples from ALL policies (need all for Fisher)
        all_samples = policy_manager.generate_samples(
            num_samples_per_policy=T,
            prompt_prefix=args.prompt_prefix,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Extract texts and log_probs
        all_texts = [[text for text, _ in samples_q] for samples_q in all_samples]
        all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

        # Step 2: Compute CLIP embeddings for all K×T prompts
        all_embeddings = []
        for q in range(K):
            embeddings_q = embed_prompts_with_clip_text(all_texts[q], embedder)
            all_embeddings.append(embeddings_q)

        # Step 3: Compute Fisher values and gradient for ONE policy only
        primary_device = policy_manager.device
        fisher_values = []
        weighted_log_prob = torch.tensor(0.0, device=primary_device)

        for t in range(T):
            # Fisher_t from K embeddings at time t
            embeddings_t = [[all_embeddings[q][t]] for q in range(K)]
            with torch.no_grad():
                L_t = fisher_objective.compute_objective(embeddings_t)
            fisher_values.append(L_t.item())

            # Only use log_prob from the policy we're updating!
            log_prob_t = all_log_probs[policy_to_update][t].to(primary_device)

            # Accumulate: L_t · log_prob_t (only for policy_to_update)
            weighted_log_prob = weighted_log_prob + L_t * log_prob_t

        # Statistics
        fisher_arr = torch.tensor(fisher_values)
        avg_fisher = fisher_arr.mean().item()
        std_fisher = fisher_arr.std().item()

        # Gradient computation for the single policy
        policy_params = list(policy_manager.policies[policy_to_update].get_trainable_parameters())
        grads = torch.autograd.grad(weighted_log_prob, policy_params)

        # Apply gradients to ONLY the selected policy
        optimizers[policy_to_update].zero_grad()
        for param, g in zip(policy_params, grads):
            param.grad = -g / T  # Negative for ascent
        optimizers[policy_to_update].step()

        # Record history
        history["iterations"].append(iteration)
        history["objectives"].append(avg_fisher)
        history["updated_policy"].append(policy_to_update)

        # Logging
        if iteration % args.log_interval == 0 or iteration == args.num_iterations - 1:
            logger.info(
                f"Iter {iteration:3d}/{args.num_iterations} | "
                f"Updated: policy {policy_to_update} | "
                f"Fisher={avg_fisher:9.4f} (std={std_fisher:.4f})"
            )

    # Save results
    logger.info("\n--- Saving Results ---")

    policy_path = os.path.join(args.output_dir, "policies.pt")
    state = {
        "policies": [p.model.state_dict() for p in policy_manager.policies],
        "args": vars(args),
        "history": history,
    }
    torch.save(state, policy_path)
    logger.info(f"Saved policies to {policy_path}")

    # Generate final prompts
    logger.info("\n--- Generating Final Prompts ---")
    final_prompts = []
    for q, policy in enumerate(policy_manager.policies):
        prompts_q = []
        for _ in range(5):
            with torch.no_grad():
                text, _ = policy.generate_with_log_prob(
                    prompt_prefix=args.prompt_prefix,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                )
            prompts_q.append(text)
        final_prompts.append(prompts_q)
        logger.info(f"Policy {q}: {prompts_q[0][:50]}...")

    # Save prompts
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
    logger.info("Alternating Optimization Complete!")
    logger.info(f"Initial Fisher: {history['objectives'][0]:.4f}")
    logger.info(f"Final Fisher: {history['objectives'][-1]:.4f}")
    logger.info(f"Improvement: {history['objectives'][-1] - history['objectives'][0]:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
