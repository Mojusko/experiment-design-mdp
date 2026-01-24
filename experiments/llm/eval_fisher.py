#!/usr/bin/env python3
"""Quick script to evaluate Fisher objective on prompts."""

import torch
import sys
sys.path.insert(0, '.')

from components.embedder import CLIPEmbedder

def compute_fisher_objective(prompts, embedder, lambda_reg=0.01):
    """Compute D-optimal (logdet) objective for K prompts."""
    # Get embeddings
    embeddings = []
    for p in prompts:
        emb = embedder.embed_text(p)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)  # (K, d)
    K, d = embeddings.shape

    # Fisher = (1/K) * sum(phi_i @ phi_i.T) - mu @ mu.T + lambda * I
    mu = embeddings.mean(dim=0)
    second_moment = (embeddings.T @ embeddings) / K
    fisher = second_moment - torch.outer(mu, mu)
    fisher = fisher + lambda_reg * torch.eye(d, device=fisher.device)

    # D-optimal: logdet
    sign, logdet = torch.linalg.slogdet(fisher)
    return logdet.item()

def main():
    print("Loading CLIP embedder...")
    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        normalize=True,
    )

    # Final prompts from experiment (with EOS collapse)
    original_prompts = [
        "<|endoftext|>",  # Policy 0 - collapsed
        "a very beautiful anime girl, full body, long braided curly silver hair, sky blue eyes",
        "<|endoftext|>",  # Policy 2 - collapsed
        "a painting of a man with his face painted like a gas mask Trending on",
    ]

    # Fixed prompts - replace EOS with random SD prompts
    fixed_prompts = [
        "a majestic dragon flying over mountains, fantasy art, highly detailed, artstation",  # Replace Policy 0
        "a very beautiful anime girl, full body, long braided curly silver hair, sky blue eyes",
        "cyberpunk city at night, neon lights, rain, cinematic lighting, 8k",  # Replace Policy 2
        "a painting of a man with his face painted like a gas mask Trending on",
    ]

    print("\n=== Original prompts (with EOS collapse) ===")
    for i, p in enumerate(original_prompts):
        print(f"  Policy {i}: {p[:60]}...")

    obj_original = compute_fisher_objective(original_prompts, embedder)
    print(f"\nFisher objective (logdet): {obj_original:.4f}")

    print("\n=== Fixed prompts (EOS replaced) ===")
    for i, p in enumerate(fixed_prompts):
        print(f"  Policy {i}: {p[:60]}...")

    obj_fixed = compute_fisher_objective(fixed_prompts, embedder)
    print(f"\nFisher objective (logdet): {obj_fixed:.4f}")

    print(f"\n=== Improvement: {obj_fixed - obj_original:.4f} ===")

if __name__ == "__main__":
    main()
