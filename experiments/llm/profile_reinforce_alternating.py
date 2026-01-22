#!/usr/bin/env python3
"""
Profile REINFORCE alternating algorithm - detailed timing for all processing steps.
"""

import os
import sys
import time
import torch
import warnings
from collections import defaultdict

# Suppress the padding warning
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.gpt2_generative_policy import MultiPolicyManager
from components.monte_carlo_fisher import MonteCarloFisherObjective, embed_prompts_with_clip_text
from components.embedder import CLIPEmbedder


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self):
        self.times = defaultdict(list)
        self.current_name = None
        self.start_time = None

    def __call__(self, name):
        self.current_name = name
        return self

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self.start_time
        self.times[self.current_name].append(elapsed)

    def summary(self):
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        total = 0
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            total_time = sum(times)
            total += total_time
            print(f"{name:40s}: {avg*1000:8.2f} ms avg, {total_time:8.3f}s total ({len(times)} calls)")
        print("-" * 70)
        print(f"{'TOTAL':40s}: {total:8.3f}s")
        print("=" * 70)


def main():
    # Configuration
    NUM_ITERATIONS = 100
    K = 4  # Number of policies
    T = 50  # Samples per iteration
    MAX_NEW_TOKENS = 15
    TEMPERATURE = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DESIGN = "A"  # A-optimal design: maximize -tr(I^{-1})
    LOG_INTERVAL = 20

    timer = Timer()

    print("=" * 70)
    print(f"REINFORCE ALTERNATING - {DESIGN}-OPTIMAL DESIGN")
    print("=" * 70)
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Policies (K): {K}")
    print(f"Samples per iteration (T): {T}")
    print(f"Total samples per iteration: K×T = {K*T}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Design criterion: {DESIGN}-optimal")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Initialize components
    print("\n--- Initialization ---")

    with timer("init_policy_manager"):
        policy_manager = MultiPolicyManager(
            num_policies=K,
            model_name="gpt2",
            lora_rank=8,
            lora_alpha=16.0,
            device=DEVICE,
        )

    with timer("init_fisher_objective"):
        fisher_objective = MonteCarloFisherObjective(
            lambda_reg=0.1,
            device=DEVICE,
        )

    with timer("init_clip_embedder"):
        embedder = CLIPEmbedder(
            model_id="openai/clip-vit-large-patch14",
            normalize=True,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        )

    # Create optimizers
    optimizers = []
    for q in range(K):
        opt = torch.optim.AdamW(
            policy_manager.policies[q].get_trainable_parameters(),
            lr=1e-4,
        )
        optimizers.append(opt)

    print("\n--- Starting Optimization ---")
    iteration_times = []
    objective_history = []  # Track objective at each iteration

    # Print table header
    print("\n" + "=" * 50)
    print(f"{'Iter':>6} | {'Objective (A-opt)':>18} | {'Time (s)':>10}")
    print("=" * 50)

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.perf_counter()
        policy_to_update = iteration % K

        # Step 1: Generate T samples from ALL K policies
        with timer("generate_samples"):
            all_samples = policy_manager.generate_samples(
                num_samples_per_policy=T,
                prompt_prefix="",
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )

        # Extract texts and log_probs
        all_texts = [[text for text, _ in samples_q] for samples_q in all_samples]
        all_log_probs = [[log_prob for _, log_prob in samples_q] for samples_q in all_samples]

        # Step 2: Compute CLIP embeddings for all K×T prompts
        with timer("clip_embedding"):
            all_embeddings = []
            for q in range(K):
                embeddings_q = embed_prompts_with_clip_text(all_texts[q], embedder)
                all_embeddings.append(embeddings_q)

        # Step 3: Compute Fisher values for all T samples (A-optimal design)
        primary_device = policy_manager.device
        fisher_values = []
        weighted_log_prob = torch.tensor(0.0, device=primary_device)

        with timer("fisher_computation"):
            for t in range(T):
                # Fisher_t from K embeddings at time t
                embeddings_t = [[all_embeddings[q][t]] for q in range(K)]
                with torch.no_grad():
                    L_t = fisher_objective.compute_objective(embeddings_t, design=DESIGN)
                fisher_values.append(L_t.item())

                log_prob_t = all_log_probs[policy_to_update][t].to(primary_device)
                weighted_log_prob = weighted_log_prob + L_t * log_prob_t

        # Step 4: Gradient computation
        with timer("gradient_computation"):
            policy_params = list(policy_manager.policies[policy_to_update].get_trainable_parameters())
            grads = torch.autograd.grad(weighted_log_prob, policy_params)

        # Step 5: Optimizer step
        with timer("optimizer_step"):
            optimizers[policy_to_update].zero_grad()
            for param, g in zip(policy_params, grads):
                param.grad = -g / T
            optimizers[policy_to_update].step()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        iter_time = time.perf_counter() - iter_start
        iteration_times.append(iter_time)

        # Stats
        avg_objective = sum(fisher_values) / len(fisher_values)
        objective_history.append(avg_objective)

        # Print every LOG_INTERVAL iterations
        if iteration % LOG_INTERVAL == 0 or iteration == NUM_ITERATIONS - 1:
            print(f"{iteration:>6} | {avg_objective:>18.6f} | {iter_time:>10.2f}")

    print("=" * 50)

    # Print summary table of objectives every 20 iterations
    print("\n" + "=" * 60)
    print(f"OBJECTIVE VALUES SUMMARY ({DESIGN}-OPTIMAL)")
    print("=" * 60)
    print(f"{'Iteration':>10} | {'Objective':>15} | {'Change':>15}")
    print("-" * 60)

    prev_obj = None
    for i in range(0, NUM_ITERATIONS, LOG_INTERVAL):
        obj = objective_history[i]
        if prev_obj is not None:
            change = obj - prev_obj
            change_str = f"{change:+15.6f}"
        else:
            change_str = "           N/A"
        print(f"{i:>10} | {obj:>15.6f} | {change_str}")
        prev_obj = obj

    # Final iteration if not already printed
    if (NUM_ITERATIONS - 1) % LOG_INTERVAL != 0:
        obj = objective_history[-1]
        change = obj - prev_obj
        print(f"{NUM_ITERATIONS-1:>10} | {obj:>15.6f} | {change:+15.6f}")

    print("-" * 60)
    print(f"{'Initial':>10} | {objective_history[0]:>15.6f}")
    print(f"{'Final':>10} | {objective_history[-1]:>15.6f}")
    print(f"{'Total Δ':>10} | {objective_history[-1] - objective_history[0]:>+15.6f}")
    print("=" * 60)

    # Print timing summary
    timer.summary()

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Total iterations: {NUM_ITERATIONS}")
    print(f"Average iteration time: {sum(iteration_times)/len(iteration_times):.3f}s")
    print(f"Total time: {sum(iteration_times):.1f}s ({sum(iteration_times)/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
