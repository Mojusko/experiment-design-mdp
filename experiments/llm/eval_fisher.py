#!/usr/bin/env python3
"""Quick script to evaluate Fisher objective on prompts."""

import torch
from transformers import CLIPModel, CLIPProcessor

def get_clip_embedding(text, model, processor, device="cuda"):
    """Get CLIP text embedding."""
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
    return emb.squeeze(0)

def compute_fisher_objective(prompts, model, processor, device="cuda", lambda_reg=0.01):
    """Compute D-optimal (logdet) objective for K prompts."""
    # Get embeddings
    embeddings = []
    for p in prompts:
        emb = get_clip_embedding(p, model, processor, device)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

    obj_original = compute_fisher_objective(original_prompts, model, processor, device)
    print(f"\nFisher objective (logdet): {obj_original:.4f}")

    print("\n=== Fixed prompts (EOS replaced) ===")
    for i, p in enumerate(fixed_prompts):
        print(f"  Policy {i}: {p[:60]}...")

    obj_fixed = compute_fisher_objective(fixed_prompts, model, processor, device)
    print(f"\nFisher objective (logdet): {obj_fixed:.4f}")

    print(f"\n=== Improvement: {obj_fixed - obj_original:.4f} ===")

if __name__ == "__main__":
    main()
