# Proposed Solution: LLM-Based Experimental Design (Vocabulary-Free)

This document specifies how to use GPT-2 for vocabulary-free experimental design, eliminating the fixed vocabulary constraint of ED-PBRL.

---

## 1. The Goal

**Eliminate the fixed vocabulary.** Let GPT-2 generate any prompt it wants.

| ED-PBRL (Current) | Our Approach |
|-------------------|--------------|
| Fixed vocab S = {word₁, ..., word_{2500}} | No fixed vocab — GPT-2's full vocabulary |
| Explicit d ∈ ℝ^{2500} | Implicit p_θ(prompt) via LLM |
| Pre-computed Φ matrix | On-the-fly φ(prompt) via CLIP |
| Frank-Wolfe (closed-form) | REINFORCE (sampling-based) |

---

## 2. Architecture

### 2.1 The Pipeline

**During Optimization (fast, no image generation):**
```
GPT-2 with parameters θ_q
    ↓
Sample prompt: τ ~ p_{θ_q}  (autoregressive generation)
    ↓
Decode: τ = "A cyberpunk city at night"
    ↓
CLIP text encoder: φ(τ) = CLIP_text(τ) ∈ ℝ^768   ← Fast!
    ↓
Fisher Information: computed from {φ(τ)}
    ↓
REINFORCE: ∇_{θ_q} objective
```

**After Optimization (for feedback collection):**
```
Optimized GPT-2 policies
    ↓
Generate prompts for exploration
    ↓
Stable Diffusion: images = SD(prompts)   ← Only here!
    ↓
Show to users, collect preferences
    ↓
Estimate θ̂ from feedback
```

**Key insight**: CLIP text and image embeddings are aligned (that's CLIP's design).
Using text embeddings during optimization is fast and a good proxy for the final
image-based Fisher Information.

### 2.2 No Vocabulary Projection

**Critical**: There is NO projection layer mapping GPT-2's vocabulary to a fixed vocabulary.

GPT-2 outputs a distribution over its native ~50k BPE tokens. We sample from this distribution to generate complete prompts. The prompts are then embedded by CLIP.

---

## 3. The Objective

### 3.1 Fisher Information (Monte Carlo Estimate)

For K policies θ₁, ..., θ_K, sample N prompts from each:

$$\tau_q^{(1)}, ..., \tau_q^{(N)} \sim p_{\theta_q}$$

Compute CLIP text embeddings (fast, no image generation needed):

$$\phi_q^{(j)} = \text{CLIP}_{\text{text}}(\tau_q^{(j)})$$

Estimate moments:

$$\hat{\mu}_q = \frac{1}{N} \sum_{j=1}^{N} \phi_q^{(j)}$$

$$\hat{\Sigma}_q = \frac{1}{N} \sum_{j=1}^{N} \phi_q^{(j)} (\phi_q^{(j)})^\top$$

Fisher Information:

$$\hat{I} = \frac{1}{K} \sum_{q=1}^{K} \hat{\Sigma}_q - \bar{\mu} \bar{\mu}^\top + \lambda I_d$$

where $\bar{\mu} = \frac{1}{K} \sum_q \hat{\mu}_q$.

Objective (D-optimal):

$$\mathcal{L} = \log \det(\hat{I})$$

### 3.2 Gradient via REINFORCE

We cannot backpropagate through discrete token sampling. Use the REINFORCE estimator:

$$\nabla_{\theta_q} \mathbb{E}[\mathcal{L}] = \mathbb{E}\left[ \mathcal{L} \cdot \sum_{j=1}^{N} \nabla_{\theta_q} \log p_{\theta_q}(\tau_q^{(j)}) \right]$$

The Monte Carlo estimate:

$$\nabla_{\theta_q} \mathbb{E}[\mathcal{L}] \approx \mathcal{L} \cdot \frac{1}{N} \sum_{j=1}^{N} \nabla_{\theta_q} \log p_{\theta_q}(\tau_q^{(j)})$$

All prompts from policy $q$ are weighted by the same global objective $\mathcal{L}$.

---

## 4. The Log-Probability Gradient

GPT-2 generates autoregressively:

$$p_{\theta}(\tau) = \prod_{i=1}^{|\tau|} p_{\theta}(\text{token}_i | \text{token}_{1:i-1})$$

So:

$$\log p_{\theta}(\tau) = \sum_{i=1}^{|\tau|} \log p_{\theta}(\text{token}_i | \text{token}_{1:i-1})$$

And:

$$\nabla_{\theta} \log p_{\theta}(\tau) = \sum_{i=1}^{|\tau|} \nabla_{\theta} \log p_{\theta}(\text{token}_i | \text{token}_{1:i-1})$$

**The gradient decomposes over tokens.** Each token position contributes to the overall gradient.

---

## 5. The Two Granularities (Key Insight)

| What | Computed At |
|------|-------------|
| φ(τ) — CLIP embedding | Complete prompt level |
| ∇log p_θ(τ) — policy gradient | Token level (summed) |

**These do not need to match.** REINFORCE broadcasts the global objective to all prompts and tokens.

Example:
- Prompt: "A serene mountain landscape at golden hour"
- Tokens: ["A", " ser", "ene", " mountain", " landscape", " at", " golden", " hour"]
- φ(τ): Single 768-dim vector from CLIP
- L: Scalar objective computed from all K×N embeddings
- Gradient: L multiplied by ∇log p(τ) = Σ_t ∇log p(token_t | prefix)

**All prompts and tokens are weighted by the same global objective L.** This is standard REINFORCE.

---

## 6. Algorithm

```python
# Initialize K GPT-2 policies (e.g., with different LoRA adapters)
policies = [GPT2WithLoRA(base_model, lora_params_q) for q in range(K)]
optimizer = AdamW(all_lora_params, lr=1e-4)
clip_text_encoder = CLIPTextEncoder()  # Fast text embedding

for iteration in range(num_iterations):
    all_embeddings = []  # List of K lists of N embeddings
    all_log_probs = []   # List of K lists of N log-probs

    # Step 1: Sample prompts from each policy
    for q, policy in enumerate(policies):
        embeddings_q = []
        log_probs_q = []

        for j in range(N):
            # Generate prompt autoregressively
            prompt_tokens, log_prob = policy.generate_with_log_prob(max_length=20)
            prompt_text = tokenizer.decode(prompt_tokens)

            # Compute CLIP TEXT embedding (fast! no SD needed)
            phi = clip_text_encoder(prompt_text)

            embeddings_q.append(phi)
            log_probs_q.append(log_prob)

        all_embeddings.append(embeddings_q)
        all_log_probs.append(log_probs_q)

    # Step 2: Compute Fisher Information (Monte Carlo)
    I_hat = compute_fisher_info(all_embeddings, lambda_reg)
    L = torch.linalg.slogdet(I_hat)[1]  # log-det objective

    # Step 3: REINFORCE loss
    # loss = -L * (1/N) * sum_j log p(τ_j)  for each policy
    total_log_prob = sum(
        log_prob for log_probs_q in all_log_probs for log_prob in log_probs_q
    ) / (K * N)
    loss = -L * total_log_prob  # Negative because we maximize

    # Step 4: Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# AFTER OPTIMIZATION: Generate images for feedback collection
final_prompts = [policy.generate(temperature=0.7) for policy in policies]
images = [stable_diffusion(prompt) for prompt in final_prompts]  # SD only here!
# Collect user preferences on images, then estimate θ̂
```

---

## 7. Implementation Considerations

### 7.1 Prompt Generation

Use GPT-2's `generate()` with:
- `do_sample=True` for stochastic generation
- `max_new_tokens=20` (or desired prompt length)
- `return_dict_in_generate=True, output_scores=True` to get log-probs

### 7.2 LoRA for Efficiency

Fine-tune GPT-2 with LoRA (Low-Rank Adaptation):
- ~1M parameters per policy instead of 124M
- Shared base model, different adapters for each policy

### 7.3 Variance Reduction

REINFORCE has high variance. Mitigate with:
- Multiple samples per policy (N = 16-64)
- Gradient clipping
- Optional: subtract baseline from L (e.g., moving average of past objectives)

### 7.4 Computational Cost

**During REINFORCE optimization**: Very fast!
- CLIP text encoding is ~100x faster than SD image generation
- K policies × N samples × CLIP_text inference
- For K=4, N=16: 64 text embeddings per iteration (milliseconds)
- No SD during optimization

**After optimization**: SD only for final prompt generation
- Generate prompts from trained policies
- Run SD once per final prompt for feedback collection
- Much fewer images than if SD were in the loop

---

## 8. What We Do NOT Do

**NO vocabulary projection.** We do not map GPT-2's outputs to a fixed vocabulary.

**NO pre-computed embeddings.** We compute CLIP embeddings on-the-fly for each generated prompt.

**NO Frank-Wolfe.** Convex optimization doesn't apply when the policy is a neural network.

**NO per-token embeddings.** CLIP embeds the complete prompt (via generated image), not individual tokens.

---

## 9. Comparison with Previous (Wrong) Approach

| Previous (Wrong) | Correct Approach |
|------------------|------------------|
| Project GPT-2 output to fixed vocab | Use GPT-2's native vocabulary |
| Explicit d ∈ ℝ^{2500} | Sample τ ~ p_θ(prompt) |
| Reuse MultiPolicyOrigDesignD | Monte Carlo Fisher estimation |
| Direct backprop through softmax | REINFORCE through sampling |

The previous approach tried to fit GPT-2 into the existing ED-PBRL infrastructure. This defeated the purpose — we were still constrained to a fixed vocabulary.

---

## 10. Summary

### The Pipeline

**During Optimization (fast, CLIP text only):**
```
θ_q → GPT-2 → sample prompt τ → CLIP_text → φ(τ) ∈ ℝ^768
                                               ↓
                                   Fisher Info (Monte Carlo)
                                               ↓
                                   REINFORCE → ∇_{θ_q}
                                               ↓
                                   Update LoRA weights
```

**After Optimization (for feedback collection):**
```
Trained θ_q → GPT-2 → generate prompts → SD → images
                                                  ↓
                                        Show to users
                                                  ↓
                                        Collect preferences
                                                  ↓
                                        Estimate θ̂
```

### The Key Equations

**Fisher Information (estimated):**
$$\hat{I} = \frac{1}{K} \sum_q \hat{\Sigma}_q - \bar{\mu}\bar{\mu}^\top + \lambda I_d$$

**Objective:**
$$\mathcal{L} = \log \det(\hat{I})$$

**REINFORCE gradient:**
$$\nabla_{\theta_q} \mathbb{E}[\mathcal{L}] \approx \mathcal{L} \cdot \frac{1}{N} \sum_{j=1}^{N} \nabla_{\theta_q} \log p_{\theta_q}(\tau_q^{(j)})$$

**Log-prob decomposition:**
$$\log p_{\theta}(\tau) = \sum_{t=1}^{|\tau|} \log p_{\theta}(w_t | w_{1:t-1})$$

---

## 11. Implementation Findings (January 2025)

### 11.1 Word-Level Intermediate Embeddings

**Key enhancement**: Instead of embedding only the final prompt, embed at each word position:

```
Prompt: "a cyberpunk city at night glowing neon"
        ↓
Prefixes: ["a", "a cyberpunk", "a cyberpunk city", ..., full prompt]
        ↓
Embeddings: φ₁, φ₂, ..., φ_H  (H embeddings per prompt)
```

This gives K×H embeddings per iteration instead of just K, providing richer gradient signal.

### 11.2 Best Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| λ (regularization) | 0.5 | Smooth trajectory; paper uses λ=100 but scales differ |
| Learning rate | 1e-6 | For SGD; Adam needs ~1e-5 |
| Optimizer | SGD | Faster convergence than Adam for this problem |
| Baseline | none | Simple L×Σlogprob; fastest convergence |
| Init noise | 0 | Start identical; let optimization create diversity |
| M (samples) | 10 | Fisher samples per iteration |
| H (horizon) | 8 | Words per prompt |

### 11.3 `<|endoftext|>` Handling

MagicPrompt expects input prefixes, but we use it prefix-free. Special handling:
1. **Block at start**: Mask `<|endoftext|>` logit for first 3 tokens
2. **Allow natural endings**: After 3 tokens, allow EOS for proper completion
3. **Strip for embedding**: Remove `<|endoftext|>` before CLIP embedding

### 11.4 Model Choice

**MagicPrompt** (GPT-2 fine-tuned on 80k Lexica.art prompts) works better than base GPT-2 for generating Stable Diffusion prompts.

---

## References

- **ED-PBRL**: Schacht et al. (2025), arXiv:2512.19057v1
- **REINFORCE**: Williams (1992)
- **GPT-2**: Radford et al. (2019)
- **LoRA**: Hu et al. (2021)
- **CLIP**: Radford et al. (2021)
- **MagicPrompt**: Gustavosta/MagicPrompt-Stable-Diffusion (HuggingFace)

---

*This document specifies the vocabulary-free approach using LLMs and REINFORCE.*
