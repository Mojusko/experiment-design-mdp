"""
GPT-2 Generative Policy for REINFORCE-based experimental design.

This module implements vocabulary-free prompt generation using GPT-2 with LoRA adapters.
Each policy learns to generate prompts that maximize Fisher Information.

Key insight: We sample complete sentences from GPT-2, compute phi via SD+CLIP,
and use REINFORCE to backpropagate through the sampling process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GPT2GenerativePolicy(nn.Module):
    """
    A policy that generates prompts using GPT-2 with LoRA fine-tuning.

    The base GPT-2 is frozen; only LoRA adapter parameters are trained.

    Usage:
        policy = GPT2GenerativePolicy(...)
        text, log_prob = policy.generate_with_log_prob("A photo of")

        # For REINFORCE:
        # reward = compute_reward(text)  # e.g., Fisher Info contribution
        # loss = -reward * log_prob
        # loss.backward()
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        """
        Initialize GPT-2 policy with LoRA adapter.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "gpt2-medium")
            lora_rank: LoRA rank (lower = fewer parameters, less expressive)
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to (default: c_attn, c_proj)
            device: Device to place model on
        """
        super().__init__()
        self.device = device
        self.model_name = model_name

        # Default target modules for GPT-2
        if target_modules is None:
            target_modules = ["c_attn", "c_proj"]

        # Load base model
        logger.info(f"Loading GPT-2 model: {model_name}")
        base_model = GPT2LMHeadModel.from_pretrained(model_name)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move to device
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_with_log_prob(
        self,
        prompt_prefix: str = "",
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        stop_at_period: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate text autoregressively and return log-probability.

        Args:
            prompt_prefix: Starting text (e.g., "A photo of")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            stop_at_period: Stop generation when a period is encountered

        Returns:
            generated_text: The complete generated text (prefix + new tokens)
            log_prob: Sum of log-probabilities of generated tokens (for REINFORCE)
        """
        # Encode prefix
        if prompt_prefix:
            input_ids = self.tokenizer.encode(prompt_prefix, return_tensors="pt").to(self.device)
        else:
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

        generated_ids = input_ids.clone()

        # Autoregressive generation
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(generated_ids)
            next_token_logits = outputs.logits[0, -1, :].clone()  # (vocab_size,)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][-1]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Stop conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            if stop_at_period:
                token_text = self.tokenizer.decode([next_token.item()])
                if '.' in token_text:
                    break

        # Now compute log-probs WITH gradients
        # Re-run forward pass on the complete sequence
        full_outputs = self.model(generated_ids)
        full_logits = full_outputs.logits  # (1, seq_len, vocab_size)

        # Compute log-prob for each generated token
        prefix_len = input_ids.shape[1]
        total_log_prob = torch.tensor(0.0, device=self.device)

        for i in range(prefix_len, generated_ids.shape[1]):
            # Logits at position i-1 predict token at position i
            token_logits = full_logits[0, i - 1, :] / temperature

            # Apply same filtering as during generation for consistency
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, token_logits.size(-1))
                indices_to_remove = token_logits < torch.topk(token_logits, top_k_val)[0][-1]
                token_logits = token_logits.clone()
                token_logits[indices_to_remove] = float('-inf')

            log_probs = F.log_softmax(token_logits, dim=-1)
            token_id = generated_ids[0, i]
            total_log_prob = total_log_prob + log_probs[token_id]

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text, total_log_prob

    def generate_batch_with_log_probs(
        self,
        prompt_prefix: str = "",
        num_samples: int = 16,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate multiple samples in a single batched forward pass.

        Much faster than calling generate_with_log_prob() multiple times.

        Args:
            prompt_prefix: Starting text (e.g., "A photo of")
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate per sample
            temperature: Sampling temperature

        Returns:
            texts: List of generated texts
            log_probs: List of log-probability tensors (with gradients)
        """
        # Encode prefix
        if prompt_prefix:
            input_ids = self.tokenizer.encode(prompt_prefix, return_tensors="pt").to(self.device)
        else:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

        # Expand to batch
        input_ids = input_ids.expand(num_samples, -1)
        prefix_len = input_ids.shape[1]

        # Use HuggingFace generate() for efficient batched generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,  # We'll compute log probs separately
            )
        generated_ids = outputs.sequences  # (batch, seq_len)

        # Compute log-probs WITH gradients for each sample
        # Run forward pass on all generated sequences
        full_outputs = self.model(generated_ids)
        full_logits = full_outputs.logits / temperature  # (batch, seq_len, vocab_size)

        texts = []
        log_probs = []

        for b in range(num_samples):
            # Find actual length (before padding)
            seq = generated_ids[b]
            eos_positions = (seq == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0 and eos_positions[0] > prefix_len:
                actual_len = eos_positions[0].item() + 1
            else:
                actual_len = seq.shape[0]

            # Compute log-prob for generated tokens
            total_log_prob = torch.tensor(0.0, device=self.device)
            for i in range(prefix_len, actual_len):
                token_logits = full_logits[b, i - 1, :]
                lp = F.log_softmax(token_logits, dim=-1)
                token_id = seq[i]
                total_log_prob = total_log_prob + lp[token_id]

            # Decode text
            text = self.tokenizer.decode(seq[:actual_len], skip_special_tokens=True)
            texts.append(text)
            log_probs.append(total_log_prob)

        return texts, log_probs

    def compute_log_prob(self, text: str, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute log-probability of a given text under this policy.

        Useful for evaluating existing prompts.

        Args:
            text: Text to compute log-probability for
            temperature: Temperature used during generation

        Returns:
            Log-probability (scalar tensor with gradients)
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits / temperature  # (1, seq_len, vocab_size)

        # Compute log-prob for each position (predict next token)
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)  # (seq_len-1, vocab_size)
        target_ids = input_ids[0, 1:]  # (seq_len-1,)

        # Gather log-probs of actual tokens
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        return token_log_probs.sum()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable (LoRA) parameters for optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


class MultiPolicyManager:
    """
    Manages K GPT-2 policies, each with its own LoRA adapter.

    Memory: K copies of GPT-2 base (~500MB each) + K LoRA adapters (~1MB each).
    Note: PyTorch may share memory for frozen weights across instances.
    """

    def __init__(
        self,
        num_policies: int = 4,
        model_name: str = "gpt2",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        device: str = "cuda",
        multi_gpu: bool = True,
    ):
        """
        Initialize K policies.

        Args:
            num_policies: Number of policies (K)
            model_name: GPT-2 model name
            lora_rank: LoRA rank for each policy
            lora_alpha: LoRA alpha for each policy
            device: Device to place models on (used if multi_gpu=False)
            multi_gpu: If True and multiple GPUs available, distribute policies across GPUs
        """
        self.num_policies = num_policies

        # Determine devices for each policy
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if multi_gpu and num_gpus > 1:
            # Distribute policies across available GPUs
            self.devices = [f"cuda:{i % num_gpus}" for i in range(num_policies)]
            logger.info(f"Multi-GPU mode: distributing {num_policies} policies across {num_gpus} GPUs")
        else:
            self.devices = [device] * num_policies
        self.device = self.devices[0]  # Primary device for compatibility

        logger.info(f"Creating {num_policies} GPT-2 policies with LoRA adapters")

        self.policies = [
            GPT2GenerativePolicy(
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                device=self.devices[q],
            )
            for q in range(num_policies)
        ]

        total_params = sum(p.num_trainable_params() for p in self.policies)
        logger.info(f"Total trainable parameters across all policies: {total_params:,}")
        logger.info(f"Policy devices: {self.devices}")

    def get_all_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters from all policies."""
        params = []
        for policy in self.policies:
            params.extend(policy.get_trainable_parameters())
        return params

    def generate_samples(
        self,
        num_samples_per_policy: int,
        prompt_prefix: str = "",
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        use_batched: bool = True,
    ) -> List[List[Tuple[str, torch.Tensor]]]:
        """
        Generate samples from all policies.

        Args:
            num_samples_per_policy: Number of samples (N) per policy
            prompt_prefix: Starting text for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_batched: If True, use batched generation (much faster)

        Returns:
            List of K lists, each containing N (text, log_prob) tuples.
            all_samples[q][j] = (text, log_prob) for policy q, sample j
        """
        all_samples = []

        for q, policy in enumerate(self.policies):
            if use_batched:
                # Batched generation - much faster!
                texts, log_probs = policy.generate_batch_with_log_probs(
                    prompt_prefix=prompt_prefix,
                    num_samples=num_samples_per_policy,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                samples_q = list(zip(texts, log_probs))
            else:
                # Sequential generation (slower, for debugging)
                samples_q = []
                for j in range(num_samples_per_policy):
                    text, log_prob = policy.generate_with_log_prob(
                        prompt_prefix=prompt_prefix,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    samples_q.append((text, log_prob))
            all_samples.append(samples_q)

        return all_samples

    def __getitem__(self, idx: int) -> GPT2GenerativePolicy:
        """Get policy by index."""
        return self.policies[idx]

    def __len__(self) -> int:
        """Number of policies."""
        return self.num_policies
