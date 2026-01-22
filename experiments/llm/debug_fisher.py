#!/usr/bin/env python3
"""Debug Fisher=0 issue with diverse LoRA init."""

import torch
import numpy as np

# Simulate the Fisher computation
def compute_fisher_logdet(embeddings, lambda_reg=1.0):
    """
    embeddings: list of K tensors, each shape (d,)
    Returns: logdet of Fisher Information matrix
    """
    K = len(embeddings)
    d = embeddings[0].shape[0]

    # Stack: (K, d)
    phi = torch.stack(embeddings)

    # Per-policy second moments (N=1, so Σ_q = φ_q φ_q^T)
    # avg_second = (1/K) * Σ_q φ_q φ_q^T
    avg_second = (phi.T @ phi) / K  # (d, d)

    # Grand mean
    mu_bar = phi.mean(dim=0)  # (d,)

    # Fisher = avg_second - mu_bar mu_bar^T + λI
    I_hat = avg_second - torch.outer(mu_bar, mu_bar)
    I_hat = I_hat + lambda_reg * torch.eye(d)

    # logdet
    sign, logdet = torch.linalg.slogdet(I_hat)

    return logdet.item(), sign.item()


# Test 1: Identical embeddings
print("=" * 60)
print("Test 1: Identical embeddings (K=4, d=768)")
d = 768
phi = torch.randn(d)
phi = phi / phi.norm()  # normalize like CLIP
embeddings = [phi.clone() for _ in range(4)]
logdet, sign = compute_fisher_logdet(embeddings, lambda_reg=1.0)
print(f"  lambda=1.0: logdet={logdet:.4f}, sign={sign}")
logdet, sign = compute_fisher_logdet(embeddings, lambda_reg=0.1)
print(f"  lambda=0.1: logdet={logdet:.4f}, sign={sign}")

# Test 2: Slightly different embeddings
print("\nTest 2: Slightly different embeddings (small noise)")
embeddings = [phi + 0.01 * torch.randn(d) for _ in range(4)]
embeddings = [e / e.norm() for e in embeddings]
logdet, sign = compute_fisher_logdet(embeddings, lambda_reg=1.0)
print(f"  lambda=1.0: logdet={logdet:.4f}, sign={sign}")

# Test 3: Random different embeddings
print("\nTest 3: Random different embeddings")
embeddings = [torch.randn(d) for _ in range(4)]
embeddings = [e / e.norm() for e in embeddings]
logdet, sign = compute_fisher_logdet(embeddings, lambda_reg=1.0)
print(f"  lambda=1.0: logdet={logdet:.4f}, sign={sign}")

# Test 4: Cosine similarity between embeddings
print("\nTest 4: Cosine similarities in Test 3")
for i in range(4):
    for j in range(i+1, 4):
        cos_sim = torch.dot(embeddings[i], embeddings[j]).item()
        print(f"  φ_{i} · φ_{j} = {cos_sim:.4f}")

# Test 5: What happens with different lambda values
print("\nTest 5: Effect of lambda on identical embeddings")
phi = torch.randn(d)
phi = phi / phi.norm()
embeddings = [phi.clone() for _ in range(4)]
for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
    logdet, sign = compute_fisher_logdet(embeddings, lambda_reg=lam)
    expected = d * np.log(lam)
    print(f"  lambda={lam:5.2f}: logdet={logdet:10.4f}, expected d*log(λ)={expected:10.4f}")

print("\n" + "=" * 60)
print("CONCLUSION: With identical embeddings and λ=1.0, logdet=0")
print("This explains Fisher=0.0000 in the diverse init experiment!")
print("=" * 60)
