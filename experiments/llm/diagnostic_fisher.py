#!/usr/bin/env python3
"""Diagnostic: Run 10 samples, report logdet(Fisher) and logprob for each."""

import torch
import logging
from components.gpt2_generative_policy import MultiPolicyManager
from components.embedder import CLIPEmbedder
from components.monte_carlo_fisher import MonteCarloFisherObjective, embed_prompts_with_clip_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    K = 4  # policies
    T = 10  # samples
    max_new_tokens = 6
    lambda_reg = 0.1

    logger.info(f"Initializing {K} policies...")
    policy_manager = MultiPolicyManager(
        num_policies=K,
        model_name="gpt2",
        lora_rank=8,
        lora_alpha=16.0,
        device="cuda",
        multi_gpu=True,
    )

    logger.info("Initializing embedder...")
    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        cache_dir="/tmp/clip_cache",
        normalize=True,
        device="cuda"
    )

    logger.info("Initializing Fisher objective...")
    embedding_dim = 768
    fisher_objective = MonteCarloFisherObjective(
        embedding_dim=embedding_dim,
        lambda_reg=lambda_reg,
        device="cuda",
    )

    logger.info(f"\nGenerating {T} samples, each with {K} prompts...\n")

    # Generate T samples from each of K policies
    # Returns: List of K lists, each containing T (text, log_prob) tuples
    all_samples = policy_manager.generate_samples(
        num_samples_per_policy=T,
        prompt_prefix="",
        max_new_tokens=max_new_tokens,
        temperature=1.0,
    )

    # Extract texts and log_probs
    all_texts = []  # all_texts[q] = list of T texts
    all_log_probs = []  # all_log_probs[q] = list of T log_probs
    for q in range(K):
        texts_q = [sample[0] for sample in all_samples[q]]
        log_probs_q = [sample[1] for sample in all_samples[q]]
        all_texts.append(texts_q)
        all_log_probs.append(log_probs_q)

    # Embed all prompts using CLIP text encoder
    logger.info("Embedding all prompts with CLIP text encoder...")
    all_embeddings = []
    for q in range(K):
        embeddings_q = embed_prompts_with_clip_text(all_texts[q], embedder)
        all_embeddings.append(embeddings_q)

    # Compute logdet for each of T samples
    print("\n" + "="*100)
    print(f"{'Sample':<8} {'logdet':<12} {'logprob':<12} {'Prompts (K=4)'}")
    print("="*100)

    for t in range(T):
        # logdet_t from K embeddings at time t
        embeddings_t = [[all_embeddings[q][t]] for q in range(K)]
        with torch.no_grad():
            L_t = fisher_objective.compute_objective(embeddings_t)

        # logprob_t = sum of K log_probs
        log_prob_t = sum(all_log_probs[q][t].item() for q in range(K))

        # Get prompts for this sample (truncate for display)
        prompts_t = [all_texts[q][t][:30] for q in range(K)]

        print(f"{t:<8} {L_t.item():<12.4f} {log_prob_t:<12.2f} {prompts_t}")

    print("="*100)

if __name__ == "__main__":
    main()
