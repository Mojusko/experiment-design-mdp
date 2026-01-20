#!/usr/bin/env python3
"""
REINFORCE with word-level intermediate embeddings.

Key idea:
- Generate prompts until H complete words (spaces as delimiter)
- Embed H prefixes per policy (clean word boundaries for CLIP)
- Fisher from K×H embeddings (much better than K=4)
- Log-probs at BPE level, aggregated per word

Multi-GPU: Each policy generates on its own GPU in parallel.
"""

import os
import sys
import time
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

warnings.filterwarnings("ignore", message=".*right-padding.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.embedder import CLIPEmbedder


class WordLevelPolicy:
    """GPT-2 policy that generates until H complete words."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda:0", seed: int = None):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from peft import get_peft_model, LoraConfig, TaskType
        import numpy as np
        import random

        self.device = device

        # Set seed for diverse LoRA init
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Load base model
        base_model = GPT2LMHeadModel.from_pretrained(model_name)

        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )

        self.model = get_peft_model(base_model, lora_config)

        # Perturb LoRA weights for diversity
        if seed is not None:
            torch.manual_seed(seed)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and 'lora' in name.lower():
                        noise = torch.randn_like(param) * 0.01  # Reduced from 0.1 to preserve coherence
                        param.add_(noise)

        self.model.to(device)

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate_until_h_words(
        self,
        h_words: int = 6,
        max_tokens: int = 30,
        temperature: float = 1.0,
        prompt_prefix: str = "",
    ) -> Tuple[str, List[float], List[int]]:
        """
        Generate tokens until we have h_words complete words (spaces as delimiter).

        Returns:
            prompt: The generated text (including prefix)
            token_logprobs: Log-prob for each generated token (with gradients)
            word_boundaries: Token indices where each word ends
        """
        # Start with prefix or BOS
        if prompt_prefix:
            input_ids = self.tokenizer.encode(prompt_prefix, return_tensors="pt").to(self.device)
            current_text = prompt_prefix
            num_spaces = prompt_prefix.count(" ")
        else:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
            current_text = ""
            num_spaces = 0

        generated_tokens = []
        token_logprobs = []
        word_boundaries = []  # Token index where each word ends

        # Count words needed after prefix
        prefix_words = len(prompt_prefix.split()) if prompt_prefix else 0
        target_spaces = prefix_words + h_words  # Total words we want

        for step in range(max_tokens):
            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :] / temperature

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Log prob (with gradient)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logprob = log_probs[next_token]

            # Decode token
            token_text = self.tokenizer.decode([next_token.item()])

            generated_tokens.append(next_token.item())
            token_logprobs.append(token_logprob)

            # Update text
            current_text += token_text

            # Count spaces (word boundaries)
            new_spaces = token_text.count(" ")
            if new_spaces > 0:
                # Mark word boundary at this token
                for _ in range(new_spaces):
                    if len(word_boundaries) < h_words:
                        word_boundaries.append(len(generated_tokens))
                num_spaces += new_spaces

            # Check if we have enough words (prefix + h_words)
            if num_spaces >= target_spaces:
                break

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # If we didn't get enough words, pad word_boundaries
        while len(word_boundaries) < h_words:
            word_boundaries.append(len(generated_tokens))

        return current_text.strip(), token_logprobs, word_boundaries[:h_words]

    def get_trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class MultiGPUPolicyManager:
    """Manages K policies across multiple GPUs."""

    def __init__(self, k: int = 4, model_name: str = "gpt2"):
        num_gpus = torch.cuda.device_count()
        self.devices = [f"cuda:{i % num_gpus}" for i in range(k)]

        print(f"Initializing {k} policies across {num_gpus} GPUs...")
        self.policies = [
            WordLevelPolicy(
                model_name=model_name,
                device=self.devices[q],
                seed=42 + q * 1000,
            )
            for q in range(k)
        ]
        print(f"Policies on devices: {self.devices}")

    def generate_all_parallel(
        self,
        h_words: int = 6,
        temperature: float = 1.0,
        prompt_prefix: str = "",
    ) -> List[Tuple[str, List, List]]:
        """Generate from all policies in parallel."""

        def gen_one(policy):
            return policy.generate_until_h_words(h_words=h_words, temperature=temperature, prompt_prefix=prompt_prefix)

        with ThreadPoolExecutor(max_workers=len(self.policies)) as executor:
            results = list(executor.map(gen_one, self.policies))

        return results


def build_word_prefixes(prompt: str, h_words: int) -> List[str]:
    """Build H prefixes at word boundaries."""
    words = prompt.split()[:h_words]
    prefixes = []
    for i in range(1, len(words) + 1):
        prefixes.append(" ".join(words[:i]))
    # Pad if needed
    while len(prefixes) < h_words:
        prefixes.append(prefixes[-1] if prefixes else "")
    return prefixes


def embed_all_prefixes_batched(
    all_prefixes: List[List[str]],  # K lists of H prefixes each
    embedder: CLIPEmbedder,
) -> torch.Tensor:
    """Batch embed all K×H prefixes."""
    # Flatten
    flat_prefixes = [p for prefixes in all_prefixes for p in prefixes]

    # Batch embed
    embeddings = []
    for prefix in flat_prefixes:
        emb = embedder.embed_text(prefix)  # (1, 768)
        embeddings.append(emb.squeeze(0))

    # Stack: (K×H, 768)
    return torch.stack(embeddings)


def compute_fisher_from_embeddings(
    embeddings: torch.Tensor,  # (K×H, d)
    k: int,
    h: int,
    lambda_reg: float = 1.0,
) -> torch.Tensor:
    """
    Compute Fisher Information from K×H embeddings.

    Structure matches ED-PBRL paper:
        I = Σ_h I_h + λI

    where I_h is the Fisher contribution at timestep h:
        I_h = (1/K) Σ_q φ_{q,h} φ_{q,h}ᵀ - μ_h μ_hᵀ
        μ_h = (1/K) Σ_q φ_{q,h}

    Each timestep h has K embeddings (one per policy).
    The total Fisher is the SUM over timesteps (not average).
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

    # Add regularization
    fisher = fisher + lambda_reg * torch.eye(d, device=device, dtype=fisher.dtype)

    return fisher


def main():
    K = 4  # policies
    H = 6  # words per prompt (after prefix)
    T = 10  # Fisher samples to average for stability
    NUM_ITERATIONS = 200
    LAMBDA_REG = 1.0
    TEMPERATURE = 1.0
    LR = 5e-6
    PROMPT_PREFIX = ""  # No prefix

    print("=" * 60)
    print("REINFORCE with Word-Level Intermediate Embeddings")
    print("=" * 60)
    print(f"K={K} policies, H={H} words per prompt, T={T} Fisher samples")
    print(f"Each Fisher from K×H = {K*H} embeddings")
    print(f"Averaged Fisher from T×K×H = {T*K*H} embeddings")
    print(f"λ={LAMBDA_REG}, lr={LR}")
    print("=" * 60)

    # Initialize
    print("\n--- Initialization ---")

    policy_manager = MultiGPUPolicyManager(k=K)

    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        normalize=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )

    # Optimizers (one per policy for alternating updates)
    optimizers = [
        torch.optim.AdamW(policy.get_trainable_parameters(), lr=LR)
        for policy in policy_manager.policies
    ]

    print("\n--- Starting Optimization ---")
    print(f"{'Iter':>5} | {'Objective':>12} | {'Time':>8} | Prompts")
    print("-" * 70)

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.perf_counter()
        policy_to_update = iteration % K

        # Step 1: Generate T×K prompts in parallel
        # Each policy generates T prompts concurrently
        def gen_t_samples(policy_idx):
            policy = policy_manager.policies[policy_idx]
            samples = []
            for _ in range(T):
                prompt, logprobs, boundaries = policy.generate_until_h_words(
                    h_words=H, temperature=TEMPERATURE, prompt_prefix=PROMPT_PREFIX
                )
                samples.append((prompt, logprobs, boundaries))
            return samples

        with ThreadPoolExecutor(max_workers=K) as executor:
            all_policy_samples = list(executor.map(gen_t_samples, range(K)))
        # all_policy_samples[q][t] = (prompt, logprobs, boundaries) for policy q, sample t

        # Compute L_t for each sample and accumulate weighted gradients
        sample_prompts = []
        L_values = []
        policy_params = list(policy_manager.policies[policy_to_update].get_trainable_parameters())
        grad_accum = [torch.zeros_like(p) for p in policy_params]

        for t in range(T):
            prompts = [all_policy_samples[q][t][0] for q in range(K)]
            all_token_logprobs = [all_policy_samples[q][t][1] for q in range(K)]

            if t == 0:
                sample_prompts = prompts

            # Step 2: Build word-level prefixes
            all_prefixes = [build_word_prefixes(p, H) for p in prompts]

            # Step 3: Embed all K×H prefixes
            embeddings = embed_all_prefixes_batched(all_prefixes, embedder)

            # Step 4: Compute Fisher_t with regularization
            fisher_t = compute_fisher_from_embeddings(embeddings, K, H, lambda_reg=LAMBDA_REG)

            # Compute objective L_t for this sample
            sign, logdet = torch.linalg.slogdet(fisher_t)
            if sign <= 0:
                print(f"Warning: Fisher_t not positive definite at iter {iteration}, sample {t}")
                continue

            L_t = logdet
            L_values.append(L_t.item())

            # Compute gradient for this sample: L_t * ∇log π_t
            logprobs_t = all_token_logprobs[policy_to_update]
            total_logprob_t = sum(logprobs_t)

            grads_t = torch.autograd.grad(total_logprob_t, policy_params, retain_graph=False)

            # Accumulate: L_t * grad / num_tokens_t
            policy_device = policy_manager.devices[policy_to_update]
            L_t_scaled = L_t.detach().to(policy_device) / len(logprobs_t)
            for i, g in enumerate(grads_t):
                grad_accum[i] = grad_accum[i] + g * L_t_scaled

        if len(L_values) == 0:
            print(f"Warning: All Fishers singular at iter {iteration}")
            continue

        # Average gradient over T samples
        optimizers[policy_to_update].zero_grad()
        for param, g_acc in zip(policy_params, grad_accum):
            param.grad = -g_acc / T  # Negative for ascent, average over T
        optimizers[policy_to_update].step()

        L = sum(L_values) / len(L_values)  # Average objective for logging

        iter_time = time.perf_counter() - iter_start

        # Log
        if iteration % 5 == 0 or iteration == NUM_ITERATIONS - 1:
            prompt_preview = sample_prompts[0][:40] + "..." if len(sample_prompts[0]) > 40 else sample_prompts[0]
            print(f"{iteration:>5} | {L:>12.4f} | {iter_time:>7.2f}s | {prompt_preview}")

    print("=" * 60)
    print("Final prompts:")
    for q, policy in enumerate(policy_manager.policies):
        prompt, _, _ = policy.generate_until_h_words(h_words=H, temperature=0.7, prompt_prefix=PROMPT_PREFIX)
        print(f"  Policy {q}: {prompt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
