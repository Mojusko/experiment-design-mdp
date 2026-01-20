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

    def generate_until_h_words_batched(
        self,
        batch_size: int = 10,
        h_words: int = 6,
        max_tokens: int = 30,
        temperature: float = 1.0,
        prompt_prefix: str = "",
    ) -> List[Tuple[str, List, List[int]]]:
        """
        Generate batch_size prompts in parallel until each has h_words complete words.

        Returns:
            List of (prompt, token_logprobs, word_boundaries) tuples
        """
        # Initialize batch
        if prompt_prefix:
            single_ids = self.tokenizer.encode(prompt_prefix, return_tensors="pt")
            input_ids = single_ids.repeat(batch_size, 1).to(self.device)
            current_texts = [prompt_prefix] * batch_size
            num_spaces = [prompt_prefix.count(" ")] * batch_size
        else:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, device=self.device)
            current_texts = [""] * batch_size
            num_spaces = [0] * batch_size

        prefix_words = len(prompt_prefix.split()) if prompt_prefix else 0
        target_spaces = prefix_words + h_words

        # Track per-sequence state
        token_logprobs_batch = [[] for _ in range(batch_size)]
        word_boundaries_batch = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for step in range(max_tokens):
            if all(finished):
                break

            # Forward pass for entire batch
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature  # (batch, vocab)

            # Sample for all sequences
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)

            # Log probs (with gradient)
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch, vocab)

            for b in range(batch_size):
                if finished[b]:
                    continue

                token_id = next_tokens[b]
                token_logprob = log_probs[b, token_id]
                token_text = self.tokenizer.decode([token_id.item()])

                token_logprobs_batch[b].append(token_logprob)
                current_texts[b] += token_text

                # Count spaces
                new_spaces = token_text.count(" ")
                if new_spaces > 0:
                    for _ in range(new_spaces):
                        if len(word_boundaries_batch[b]) < h_words:
                            word_boundaries_batch[b].append(len(token_logprobs_batch[b]))
                    num_spaces[b] += new_spaces

                # Check completion
                if num_spaces[b] >= target_spaces or token_id.item() == self.tokenizer.eos_token_id:
                    finished[b] = True

            # Append tokens for next iteration
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        # Pad word boundaries and build results
        results = []
        for b in range(batch_size):
            while len(word_boundaries_batch[b]) < h_words:
                word_boundaries_batch[b].append(len(token_logprobs_batch[b]))
            results.append((
                current_texts[b].strip(),
                token_logprobs_batch[b],
                word_boundaries_batch[b][:h_words]
            ))

        return results

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


def embed_texts_batched(
    texts: List[str],
    embedder: CLIPEmbedder,
    batch_size: int = 64,
) -> torch.Tensor:
    """Batch embed a list of texts efficiently."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Tokenize batch
        text_inputs = embedder.tokenizer(
            batch,
            padding="max_length",
            max_length=embedder.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(embedder.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            if hasattr(embedder.model, 'text_model'):
                emb = embedder.model.text_model(**text_inputs).pooler_output
            else:
                emb = embedder.model.get_text_features(**text_inputs)

        if embedder.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb)

    return torch.cat(embeddings, dim=0)


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
    LR = 1e-4  # Larger LR
    PROMPT_PREFIX = ""  # No prefix

    print("=" * 60)
    print("REINFORCE with Word-Level Intermediate Embeddings (A-optimal)")
    print("=" * 60)
    print(f"K={K} policies, H={H} words per prompt, T={T} Fisher samples")
    print(f"Each Fisher from K×H = {K*H} embeddings")
    print(f"Objective: -tr(I^{-1}) (A-optimal)")
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

        # Step 1: Generate T×K prompts in parallel (batched generation per policy)
        def gen_t_samples_batched(policy_idx):
            policy = policy_manager.policies[policy_idx]
            return policy.generate_until_h_words_batched(
                batch_size=T, h_words=H, temperature=TEMPERATURE, prompt_prefix=PROMPT_PREFIX
            )

        with ThreadPoolExecutor(max_workers=K) as executor:
            all_policy_samples = list(executor.map(gen_t_samples_batched, range(K)))
        # all_policy_samples[q][t] = (prompt, logprobs, boundaries) for policy q, sample t

        # Step 2: Collect ALL prefixes from all T samples (T×K×H total)
        all_prefixes_flat = []  # Will have T×K×H strings
        sample_prompts = [all_policy_samples[q][0][0] for q in range(K)]  # First sample for logging

        for t in range(T):
            for q in range(K):
                prompt = all_policy_samples[q][t][0]
                prefixes = build_word_prefixes(prompt, H)
                all_prefixes_flat.extend(prefixes)

        # Step 3: Batch embed ALL T×K×H prefixes at once
        all_embeddings = embed_texts_batched(all_prefixes_flat, embedder, batch_size=128)
        # Shape: (T×K×H, d)

        # Step 4: Compute L_t for each sample and accumulate weighted gradients
        L_values = []
        policy_params = list(policy_manager.policies[policy_to_update].get_trainable_parameters())
        grad_accum = [torch.zeros_like(p) for p in policy_params]

        for t in range(T):
            # Extract embeddings for sample t: K×H embeddings
            start_idx = t * K * H
            end_idx = start_idx + K * H
            embeddings_t = all_embeddings[start_idx:end_idx]

            # Compute Fisher_t with regularization
            fisher_t = compute_fisher_from_embeddings(embeddings_t, K, H, lambda_reg=LAMBDA_REG)

            # Compute A-optimal objective: L_t = -tr(Fisher_t^{-1})
            try:
                fisher_inv = torch.linalg.inv(fisher_t)
                L_t = -torch.trace(fisher_inv)
            except RuntimeError:
                print(f"Warning: Fisher_t not invertible at iter {iteration}, sample {t}")
                continue
            L_values.append(L_t.item())

            # Compute gradient for this sample: L_t * ∇log π_t
            logprobs_t = all_policy_samples[policy_to_update][t][1]
            total_logprob_t = sum(logprobs_t)

            # retain_graph=True needed since batched generation shares computation graph
            grads_t = torch.autograd.grad(total_logprob_t, policy_params, retain_graph=(t < T - 1))

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
        results = policy.generate_until_h_words_batched(batch_size=1, h_words=H, temperature=0.7, prompt_prefix=PROMPT_PREFIX)
        print(f"  Policy {q}: {results[0][0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
