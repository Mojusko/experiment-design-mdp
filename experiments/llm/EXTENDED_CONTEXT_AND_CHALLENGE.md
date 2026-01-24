# Extended Context and Challenge: Scaling ED-PBRL Beyond Fixed Vocabularies

This document articulates the core limitation of ED-PBRL and the goal of extending it to use LLMs for open-ended prompt generation.

---

## 1. The Core Limitation of ED-PBRL

### 1.1 Current Setup

ED-PBRL learns user preferences θ* ∈ ℝ^d from comparative feedback. The reward is linear in CLIP embeddings:

$$r(\tau) = (\theta^*)^\top \phi(\tau)$$

where φ(τ) is the CLIP embedding of the image generated from prompt τ.

### 1.2 The Fixed Vocabulary Constraint

**The fundamental limitation**: ED-PBRL requires a **pre-defined, finite vocabulary** S.

```
S = {"ancient", "forest", "sunny", "medieval", ...}  # ~2500 words
```

The optimization variable is an **explicit probability vector** over this vocabulary:

$$d_q \in \Delta^{|S|-1} \quad \text{(probability simplex of dimension |S|)}$$

This requires:
1. **Pre-enumeration**: List all possible tokens before optimization
2. **Pre-computation**: Compute φ(s) for every s ∈ S (store |S| × d matrix)
3. **Fixed horizon**: Prompts are exactly H tokens from the vocabulary

**Prompts are constrained to**: "word₁, word₂, ..., wordₕ" where each wordᵢ ∈ S.

### 1.3 What We Cannot Do

With a fixed vocabulary, we **cannot**:
- Generate novel phrases like "a cyberpunk samurai at dawn"
- Use compositional language beyond fixed combinations
- Explore the full space of natural language prompts
- Leverage LLM knowledge about language and imagery

---

## 2. The Goal: Vocabulary-Free Generation

### 2.1 The Vision

Replace the fixed vocabulary with an **LLM that can generate any text**:

```
Fixed vocabulary S          →    LLM (e.g., GPT-2)
Explicit d ∈ ℝ^|S|          →    Implicit distribution p_θ(text)
Pre-computed Φ matrix       →    On-the-fly CLIP embeddings
Frank-Wolfe optimization    →    Policy gradient (REINFORCE)
```

### 2.2 Key Insight: The Distribution is Over Prompts, Not Tokens

In the new formulation:
- The LLM defines a distribution over **complete prompts** (token sequences)
- Each prompt τ gets embedded: τ → SD(τ) → CLIP → φ(τ) ∈ ℝ^d
- The Fisher Information is computed from these embeddings

**There is no fixed vocabulary. The LLM explores the full space of language.**

---

## 3. Mathematical Reformulation

### 3.1 From Explicit Vectors to Implicit Distributions

| ED-PBRL (Current) | LLM-Based (Goal) |
|-------------------|------------------|
| d_q ∈ ℝ^{\|S\|} explicit vector | p_{θ_q}(τ) implicit distribution over prompts |
| Σ_s d_q(s) φ(s)φ(s)^⊤ | 𝔼_{τ ~ p_θ}[φ(τ)φ(τ)^⊤] |
| Closed-form computation | Monte Carlo estimation |
| Frank-Wolfe (convex) | REINFORCE (policy gradient) |

### 3.2 The Fisher Information in LLM Setting

For K policies with parameters θ₁, ..., θ_K:

$$\tilde{I}(\theta_{1:K}) = \frac{1}{K} \sum_{q=1}^{K} \mathbb{E}_{\tau \sim p_{\theta_q}}[\phi(\tau)\phi(\tau)^\top] - \bar{\mu}\bar{\mu}^\top$$

where:
- τ is a complete prompt sampled from LLM q
- φ(τ) = CLIP(StableDiffusion(τ)) ∈ ℝ^d
- μ_q = 𝔼_{τ ~ p_{θ_q}}[φ(τ)] is the mean embedding for policy q
- μ̄ = (1/K) Σ_q μ_q is the grand mean

### 3.3 Why REINFORCE is Necessary

The expectation is over discrete samples (text sequences). We cannot backpropagate through:
1. Discrete token sampling from the LLM
2. Text decoding
3. Image generation (Stable Diffusion)
4. CLIP encoding

**Solution**: Use REINFORCE to estimate gradients:

$$\nabla_{\theta_q} \mathbb{E}_{\tau \sim p_{\theta_q}}[f(\phi(\tau))] = \mathbb{E}_{\tau \sim p_{\theta_q}}\left[ f(\phi(\tau)) \cdot \nabla_{\theta_q} \log p_{\theta_q}(\tau) \right]$$

The LLM provides ∇log p_θ(τ) via its autoregressive log-probabilities.

---

## 4. The Two Granularities

A key insight: **φ and the gradient operate at different levels**.

| Component | Granularity |
|-----------|-------------|
| φ (CLIP embedding) | Complete prompt: "A serene mountain at sunset" |
| ∇log p_θ(τ) | Per-token: Σᵢ ∇log p_θ(tokenᵢ \| prefix) |

**This is fine.** REINFORCE broadcasts the "reward" (contribution to Fisher Info) to all tokens that generated the prompt. No intermediate embeddings needed.

---

## 5. Summary

### The Core Problem
ED-PBRL is trapped in a fixed vocabulary. It cannot explore the full richness of natural language.

### The Solution
Use an LLM (GPT-2) as the policy. The LLM generates complete prompts freely. Fisher Information is computed on CLIP embeddings of these prompts. REINFORCE provides gradients to update the LLM.

### The Key Equation

$$\nabla_{\theta_q} \text{FisherInfo} = \mathbb{E}_{\tau \sim p_{\theta_q}}\left[ R(\tau) \cdot \sum_{i=1}^{|\tau|} \nabla_{\theta_q} \log p_{\theta_q}(\text{token}_i | \text{prefix}_i) \right]$$

where R(τ) measures how prompt τ's embedding contributes to the Fisher Information.

---

## 6. Reviewer Feedback (ICLR 2025 Submission)

Key criticisms that motivate the vocabulary-free extension:

| Concern | Reviewer | Our Response |
|---------|----------|--------------|
| "State visitation measures very data hungry for high-dim" | EPjt | No state visitation—sample directly from LLM |
| "Policy extraction computationally intensive for LLMs" | GKit | Direct REINFORCE on LLM weights, no Convex-RL |
| "Simple baseline could learn distribution over vocabulary" | mNCP | LLM captures richer correlations than independent attributes |
| "Independence assumptions between design attributes" | mNCP | LLM naturally models attribute correlations |

The vocabulary-free approach directly addresses scalability concerns raised by reviewers.

---

## References

- **ED-PBRL Paper**: Schacht et al. "Efficient Personalization of Generative Models via Optimal Experimental Design" (2025). arXiv:2512.19057v1
- **REINFORCE**: Williams (1992)
- **GPT-2**: Radford et al. (2019)

---

*This document clarifies the goal: eliminate the fixed vocabulary constraint by using LLMs for open-ended generation.*
