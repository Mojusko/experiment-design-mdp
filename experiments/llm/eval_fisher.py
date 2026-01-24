#!/usr/bin/env python3
"""
Evaluate Fisher objective on prompts using word-level intermediate embeddings.

Matches the actual optimization in reinforce_word_level.py:
- Each prompt is split into H word prefixes
- K×H total embeddings used to compute Fisher
- Fisher = Σ_h I_h + λI, where I_h is per-timestep Fisher
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List


def build_word_prefixes(prompt: str, h_words: int) -> List[str]:
    """Build H prefixes at word boundaries."""
    words = prompt.split()[:h_words]
    prefixes = []
    for i in range(1, len(words) + 1):
        prefixes.append(" ".join(words[:i]))
    # Pad if needed (for short/collapsed prompts)
    while len(prefixes) < h_words:
        prefixes.append(prefixes[-1] if prefixes else prompt)
    return prefixes


def get_clip_embeddings_batched(texts: List[str], model, processor, device="cuda") -> torch.Tensor:
    """Get CLIP text embeddings for a batch of texts."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
    return emb  # (N, d)


def compute_fisher_from_embeddings(
    embeddings: torch.Tensor,  # (K×H, d)
    k: int,
    h: int,
    lambda_reg: float = 0.01,
    t_coef: float = 1.0,
) -> torch.Tensor:
    """
    Compute Fisher Information from K×H embeddings.

    Matches reinforce_word_level.py exactly:
        I = T * Σ_h I_h + λI

    where I_h is the Fisher contribution at timestep h:
        I_h = (1/K) Σ_q φ_{q,h} φ_{q,h}ᵀ - μ_h μ_hᵀ
        μ_h = (1/K) Σ_q φ_{q,h}
    """
    d = embeddings.shape[1]
    device = embeddings.device

    # Reshape to (K, H, d)
    embeddings = embeddings.view(k, h, d)

    # Compute I_h for each timestep h, then sum
    fisher = torch.zeros(d, d, device=device, dtype=embeddings.dtype)

    for t in range(h):
        # Get K embeddings at timestep t: (K, d)
        emb_t = embeddings[:, t, :]  # (K, d)

        # Mean across K policies at this timestep
        mu_t = emb_t.mean(dim=0)  # (d,)

        # Second moment: (1/K) Σ_q φ_{q,t} φ_{q,t}ᵀ
        second_t = (emb_t.T @ emb_t) / k  # (d, d)

        # I_t = second_t - μ_t μ_tᵀ
        I_t = second_t - torch.outer(mu_t, mu_t)

        # Accumulate
        fisher = fisher + I_t

    # Scale data part by T coefficient, then add regularization
    fisher = t_coef * fisher + lambda_reg * torch.eye(d, device=device, dtype=fisher.dtype)

    return fisher


def compute_fisher_objective(
    prompts: List[str],
    model,
    processor,
    device: str = "cuda",
    h_words: int = 14,
    lambda_reg: float = 0.01,
    t_coef: float = 10.0,
) -> float:
    """
    Compute D-optimal (logdet) objective for K prompts using K×H intermediate embeddings.
    """
    K = len(prompts)

    # Build all K×H prefixes
    all_prefixes = []
    for q, prompt in enumerate(prompts):
        prefixes = build_word_prefixes(prompt, h_words)
        all_prefixes.extend(prefixes)

    # Embed all K×H prefixes
    embeddings = get_clip_embeddings_batched(all_prefixes, model, processor, device)  # (K×H, d)

    # Compute Fisher
    fisher = compute_fisher_from_embeddings(embeddings, K, h_words, lambda_reg=lambda_reg, t_coef=t_coef)

    # D-optimal: logdet
    sign, logdet = torch.linalg.slogdet(fisher)
    return logdet.item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    H = 14  # words per prompt (matches optimization)
    LAMBDA_REG = 0.01
    T_COEF = 10.0

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

    print(f"\nSettings: H={H} words, λ={LAMBDA_REG}, T={T_COEF}")
    print(f"Using K×H = {len(original_prompts)}×{H} = {len(original_prompts)*H} embeddings per Fisher\n")

    print("=== Original prompts (with EOS collapse) ===")
    for i, p in enumerate(original_prompts):
        prefixes = build_word_prefixes(p, H)
        print(f"  Policy {i}: {p[:50]}...")
        print(f"           → {len(p.split())} words, padded prefixes: {len(set(prefixes))} unique")

    obj_original = compute_fisher_objective(original_prompts, model, processor, device, H, LAMBDA_REG, T_COEF)
    print(f"\nFisher objective (logdet): {obj_original:.4f}")

    print("\n=== Fixed prompts (EOS replaced) ===")
    for i, p in enumerate(fixed_prompts):
        prefixes = build_word_prefixes(p, H)
        print(f"  Policy {i}: {p[:50]}...")
        print(f"           → {len(p.split())} words, padded prefixes: {len(set(prefixes))} unique")

    obj_fixed = compute_fisher_objective(fixed_prompts, model, processor, device, H, LAMBDA_REG, T_COEF)
    print(f"\nFisher objective (logdet): {obj_fixed:.4f}")

    print(f"\n=== Improvement: {obj_fixed - obj_original:.4f} ===")


if __name__ == "__main__":
    main()
