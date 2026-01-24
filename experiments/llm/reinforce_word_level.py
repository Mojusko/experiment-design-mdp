#!/usr/bin/env python3
"""
REINFORCE with word-level intermediate embeddings.

Key idea:
- Generate prompts until H complete words (spaces as delimiter)
- Embed H prefixes per policy (clean word boundaries for CLIP)
- Fisher from K×H embeddings (much better than K=4)
- Log-probs at BPE level, aggregated per word

Multi-GPU: Each policy generates on its own GPU in parallel.

Supported base models:
- gpt2: Standard GPT-2 (default)
- microsoft/Promptist: GPT-2 fine-tuned with RL for SD prompt optimization
- Gustavosta/MagicPrompt-Stable-Diffusion: GPT-2 trained on Lexica.art prompts
"""

import os
import random
import sys
import time
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", message=".*right-padding.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Model Registry: Supported base models for prompt generation
# =============================================================================
MODEL_REGISTRY = {
    "gpt2": {
        "model_id": "gpt2",
        "tokenizer_id": "gpt2",
        "description": "Standard GPT-2 (124M params)",
    },
    "gpt2-medium": {
        "model_id": "gpt2-medium",
        "tokenizer_id": "gpt2-medium",
        "description": "GPT-2 Medium (355M params)",
    },
    "promptist": {
        "model_id": "microsoft/Promptist",
        "tokenizer_id": "gpt2",  # Promptist uses GPT-2 tokenizer
        "description": "Microsoft Promptist: GPT-2 fine-tuned with RL for SD v1.4",
    },
    "magicprompt": {
        "model_id": "Gustavosta/MagicPrompt-Stable-Diffusion",
        "tokenizer_id": "Gustavosta/MagicPrompt-Stable-Diffusion",
        "description": "MagicPrompt: GPT-2 trained on 80k Lexica.art prompts",
    },
    "distilgpt2-sd": {
        "model_id": "FredZhang7/distilgpt2-stable-diffusion",
        "tokenizer_id": "FredZhang7/distilgpt2-stable-diffusion",
        "description": "DistilGPT2 trained on 2M SD prompts (fast, lightweight)",
    },
}


def get_model_config(model_name: str) -> dict:
    """Get model configuration from registry, or treat as HuggingFace model ID."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    # Assume it's a direct HuggingFace model ID
    return {
        "model_id": model_name,
        "tokenizer_id": model_name,
        "description": f"Custom model: {model_name}",
    }

from components.embedder import CLIPEmbedder


class WordLevelPolicy:
    """GPT-2 policy that generates until H complete words.

    Supports multiple base models via MODEL_REGISTRY:
    - gpt2: Standard GPT-2
    - promptist: Microsoft Promptist (RL-tuned for SD)
    - magicprompt: MagicPrompt-Stable-Diffusion
    - distilgpt2-sd: Lightweight DistilGPT2 for SD
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda:0", seed: int = None, init_noise: float = 0.01):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import numpy as np
        import random

        self.device = device
        self.model_name = model_name
        self.seed = seed

        # Get model config from registry
        config = get_model_config(model_name)
        model_id = config["model_id"]
        tokenizer_id = config["tokenizer_id"]

        # Set seed for diverse init
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Load base model (full fine-tuning, no LoRA)
        # Use AutoModelForCausalLM for broader compatibility (Promptist, etc.)
        print(f"Loading model: {model_id} ({config['description']})")
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        # Perturb weights for diversity between policies
        if seed is not None and init_noise > 0:
            torch.manual_seed(seed)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    noise = torch.randn_like(param) * init_noise
                    param.add_(noise)

        self.model.to(device)

        # Create per-policy random generator for sampling diversity
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

        # Tokenizer - use AutoTokenizer for compatibility
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
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

            # Mask out <|endoftext|> only at the start to prevent empty generation
            # After a few tokens, allow natural endings
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and step < 3:
                logits[:, eos_id] = float('-inf')

            # Sample for all sequences (using per-policy generator for diversity)
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(-1)  # (batch,)

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
        return list(self.model.parameters())


class MultiGPUPolicyManager:
    """Manages K policies across multiple GPUs.

    Args:
        k: Number of policies
        model_name: Model identifier (see MODEL_REGISTRY for options):
            - "gpt2": Standard GPT-2
            - "promptist": Microsoft Promptist (RL-tuned for SD prompts)
            - "magicprompt": MagicPrompt-Stable-Diffusion
            - "distilgpt2-sd": Lightweight DistilGPT2 for SD
            - Or any HuggingFace model ID
    """

    def __init__(self, k: int = 4, model_name: str = "gpt2", base_seed: int = 42, init_noise: float = 0.01):
        num_gpus = torch.cuda.device_count()
        self.devices = [f"cuda:{i % num_gpus}" for i in range(k)]
        self.model_name = model_name

        config = get_model_config(model_name)
        print(f"Initializing {k} policies with: {config['description']}")
        print(f"Distributing across {num_gpus} GPUs...")

        self.policies = [
            WordLevelPolicy(
                model_name=model_name,
                device=self.devices[q],
                seed=base_seed + q * 1000,
                init_noise=init_noise,
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
    # Strip special tokens before embedding
    prompt = prompt.replace("<|endoftext|>", "").strip()
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


def compute_V_matrix(
    embeddings: torch.Tensor,  # (K×H, d)
    k: int,
    h: int,
) -> torch.Tensor:
    """
    Compute V matrix for V-design from K×H embeddings.

    At each timestep h, we have K policy embeddings.
    V_h = Σᵢⱼ (φᵢ - φⱼ)(φᵢ - φⱼ)ᵀ = 2K·(X.T @ X) - 2·(S @ S.T)

    where X is (K, d) matrix of embeddings at timestep h,
    and S = X.sum(dim=0).

    Total V = Σ_h V_h
    """
    d = embeddings.shape[1]
    device = embeddings.device

    # Reshape to (K, H, d)
    embeddings = embeddings.view(k, h, d)

    V = torch.zeros(d, d, device=device, dtype=embeddings.dtype)

    for t in range(h):
        # Get K embeddings at timestep t: (K, d)
        X_t = embeddings[:, t, :]  # (K, d)

        # X.T @ X = Σᵢ φᵢφᵢᵀ
        XTX = X_t.T @ X_t  # (d, d)

        # S = Σᵢ φᵢ
        S = X_t.sum(dim=0)  # (d,)

        # V_t = 2K·XTX - 2·S@S.T
        V_t = 2 * k * XTX - 2 * torch.outer(S, S)

        V = V + V_t

    return V


def compute_fisher_from_embeddings(
    embeddings: torch.Tensor,  # (K×H, d)
    k: int,
    h: int,
    lambda_reg: float = 1.0,
    t_coef: float = 1.0,  # T coefficient from ED-PBRL (scales data, not λ)
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

    # Scale data part by T coefficient, then add regularization (λ not scaled by T)
    fisher = t_coef * fisher + lambda_reg * torch.eye(d, device=device, dtype=fisher.dtype)

    return fisher


def compute_per_word_objectives(
    embeddings: torch.Tensor,  # (K×H, d)
    k: int,
    h: int,
    lambda_reg: float = 1.0,
    t_coef: float = 1.0,
    design: str = "D",
) -> Tuple[List[float], float]:
    """
    Compute per-word objectives L_w for proper credit assignment.

    For word w, L_w = f(S_w + λI) where S_w = Σ_{t=w}^{H-1} I_t
    This gives word w credit only for the Fisher terms it causally affects.

    Returns:
        L_w_list: List of H objective values, one per word position
        L_full: The full objective (same as L_0) for logging
    """
    d = embeddings.shape[1]
    device = embeddings.device
    dtype = embeddings.dtype

    # Reshape to (K, H, d)
    embeddings = embeddings.view(k, h, d)

    # Step 1: Compute all I_t matrices
    I_list = []
    for t in range(h):
        emb_t = embeddings[:, t, :]  # (K, d)
        mu_t = emb_t.mean(dim=0)
        second_t = (emb_t.T @ emb_t) / k
        I_t = second_t - torch.outer(mu_t, mu_t)
        I_list.append(I_t)

    # Step 2: Compute cumulative sums from the end (reverse cumsum)
    # S_w = Σ_{t=w}^{H-1} I_t
    # S_{H-1} = I_{H-1}
    # S_{H-2} = I_{H-2} + S_{H-1}
    # ...
    # S_0 = I_0 + I_1 + ... + I_{H-1} (full Fisher, minus regularization)
    S_list = [None] * h
    S_list[h - 1] = I_list[h - 1].clone()
    for w in range(h - 2, -1, -1):
        S_list[w] = I_list[w] + S_list[w + 1]

    # Step 3: Compute L_w for each word position
    reg_matrix = lambda_reg * torch.eye(d, device=device, dtype=dtype)
    L_w_list = []

    for w in range(h):
        # Fisher for word w: T * S_w + λI
        fisher_w = t_coef * S_list[w] + reg_matrix

        if design == "D":
            sign, logdet = torch.linalg.slogdet(fisher_w)
            if sign <= 0:
                L_w = torch.tensor(float('-inf'), device=device)
            else:
                L_w = logdet
        elif design == "A":
            try:
                fisher_inv = torch.linalg.inv(fisher_w)
                L_w = -torch.trace(fisher_inv)
            except RuntimeError:
                L_w = torch.tensor(float('-inf'), device=device)
        else:  # V-optimal - use full embeddings for V matrix
            try:
                V_w = compute_V_matrix(embeddings.view(k * h, d), k, h)
                fisher_inv = torch.linalg.inv(fisher_w)
                L_w = -torch.trace(V_w @ fisher_inv)
            except RuntimeError:
                L_w = torch.tensor(float('-inf'), device=device)

        L_w_list.append(L_w.item())

    # L_full = L_0 (the objective for the full trajectory)
    L_full = L_w_list[0]

    return L_w_list, L_full


def main():
    import argparse

    parser = argparse.ArgumentParser(description="REINFORCE with word-level embeddings")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Base model: gpt2, promptist, magicprompt, distilgpt2-sd, or HF model ID")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prompt prefix (e.g., 'A photo of')")
    parser.add_argument("--prefix-file", type=str, default=None,
                        help="File with prompt prefixes (one per line, randomly sampled)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
                        help="Optimizer: sgd (default, stable), adam, adamw")
    parser.add_argument("--lr", type=float, default=1e-7,
                        help="Learning rate (default: 1e-7)")
    parser.add_argument("--design", type=str, default="D", choices=["D", "A", "V"],
                        help="Design objective: D (logdet), A (-tr(I^-1)), V (-tr(V@I^-1))")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of optimization iterations (default: 20)")
    parser.add_argument("--lambda-reg", type=float, default=0.01,
                        help="Regularization lambda (default: 0.01)")
    parser.add_argument("--baseline", type=str, default="none", choices=["none", "weighted", "per-word"],
                        help="Baseline type: none (simple L*sum(logprob)), weighted (position weighting), per-word (reward-to-go)")
    parser.add_argument("-M", "--samples", type=int, default=10,
                        help="Number of Fisher samples for gradient (default: 10)")
    parser.add_argument("-H", "--horizon", type=int, default=14,
                        help="Number of words per prompt (default: 14)")
    parser.add_argument("--lr-decay", type=str, default="none", choices=["none", "step", "exponential", "cosine"],
                        help="LR decay schedule: none, step (0.5x every 50 iter), exponential (0.99x per iter), cosine")
    parser.add_argument("--no-intermediate", action="store_true",
                        help="Only embed final prompts, not intermediate prefixes (K embeddings instead of K×H)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for initialization (default: 42)")
    parser.add_argument("--init-noise", type=float, default=0.01,
                        help="Std of Gaussian noise added to weights for policy diversity (default: 0.01)")
    args = parser.parse_args()

    K = 4  # policies
    H = args.horizon  # words per prompt (after prefix)
    M = args.samples  # Fisher samples for gradient
    T = 10  # T coefficient in Fisher (scales data, not λ) - from ED-PBRL
    NUM_ITERATIONS = args.iterations
    LAMBDA_REG = args.lambda_reg
    TEMPERATURE = 1.0
    LR = args.lr  # From command line
    PROMPT_PREFIX = args.prefix  # From command line
    MODEL_NAME = args.model  # From command line
    DESIGN = args.design  # From command line
    OPTIMIZER = args.optimizer  # From command line
    BASELINE = args.baseline  # From command line
    LR_DECAY = args.lr_decay  # From command line
    NO_INTERMEDIATE = args.no_intermediate  # From command line
    BASE_SEED = args.seed  # From command line
    INIT_NOISE = args.init_noise  # From command line

    # Load prefixes from file if provided
    prefix_list = None
    if args.prefix_file:
        with open(args.prefix_file, "r") as f:
            prefix_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prefix_list)} prefixes from {args.prefix_file}")

    print("=" * 60)
    print(f"REINFORCE with Word-Level Intermediate Embeddings ({DESIGN}-optimal)")
    print("=" * 60)
    config = get_model_config(MODEL_NAME)
    print(f"Base model: {config['description']}")
    print(f"K={K} policies, H={H} words per prompt, M={M} Fisher samples")
    if NO_INTERMEDIATE:
        print(f"Each Fisher from K = {K} final embeddings (no intermediate)")
    else:
        print(f"Each Fisher from K×H = {K*H} embeddings")
    obj_desc = {"D": "logdet(I)", "A": "-tr(I^-1)", "V": "-tr(V @ I^-1)"}[DESIGN]
    print(f"Objective: {obj_desc} ({DESIGN}-optimal)")
    print(f"T={T}, λ={LAMBDA_REG}, lr={LR}, optimizer={OPTIMIZER}, baseline={BASELINE}, lr_decay={LR_DECAY}, seed={BASE_SEED}, init_noise={INIT_NOISE}")
    if PROMPT_PREFIX:
        print(f"Prompt prefix: '{PROMPT_PREFIX}'")
    elif prefix_list:
        print(f"Using {len(prefix_list)} prefixes from file (randomly sampled)")
    print("=" * 60)

    # Initialize
    print("\n--- Initialization ---")

    policy_manager = MultiGPUPolicyManager(k=K, model_name=MODEL_NAME, base_seed=BASE_SEED, init_noise=INIT_NOISE)

    embedder = CLIPEmbedder(
        model_id="openai/clip-vit-large-patch14",
        normalize=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )

    # Optimizers (one per policy for alternating updates)
    if OPTIMIZER == "sgd":
        optimizers = [
            torch.optim.SGD(policy.get_trainable_parameters(), lr=LR)
            for policy in policy_manager.policies
        ]
    elif OPTIMIZER == "adam":
        optimizers = [
            torch.optim.Adam(policy.get_trainable_parameters(), lr=LR)
            for policy in policy_manager.policies
        ]
    else:  # adamw
        optimizers = [
            torch.optim.AdamW(policy.get_trainable_parameters(), lr=LR)
            for policy in policy_manager.policies
        ]
    print(f"Optimizer: {OPTIMIZER.upper()}, LR: {LR}")

    # Learning rate schedulers
    schedulers = None
    if LR_DECAY != "none":
        if LR_DECAY == "step":
            # Halve LR every 50 iterations
            schedulers = [
                torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
                for opt in optimizers
            ]
        elif LR_DECAY == "exponential":
            # Decay by 0.99 each iteration
            schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
                for opt in optimizers
            ]
        elif LR_DECAY == "cosine":
            # Cosine annealing to 0 over all iterations
            schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_ITERATIONS)
                for opt in optimizers
            ]
        print(f"LR Scheduler: {LR_DECAY}")

    print("\n--- Starting Optimization ---")
    print(f"{'Iter':>5} | {'Objective':>12} | {'Eval Obj':>12} | {'Time':>8} | Prompts")
    print("-" * 80)

    # Evaluation settings
    T_EVAL = 10  # Number of real trajectories for evaluation
    EVAL_EVERY = 5  # Evaluate every N iterations

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.perf_counter()
        policy_to_update = iteration % K

        # === EVALUATION (every EVAL_EVERY iterations) ===
        eval_obj_str = ""
        if iteration % EVAL_EVERY == 0:
            with torch.no_grad():
                # Sample T_EVAL prompts from each policy
                def gen_eval_samples(policy_idx):
                    policy = policy_manager.policies[policy_idx]
                    return policy.generate_until_h_words_batched(
                        batch_size=T_EVAL, h_words=H, temperature=TEMPERATURE, prompt_prefix=PROMPT_PREFIX
                    )

                with ThreadPoolExecutor(max_workers=K) as executor:
                    eval_samples = list(executor.map(gen_eval_samples, range(K)))

                # Compute same objective as gradient: logdet(T × D_t + λI), averaged over T_EVAL samples
                eval_logdets = []
                for t in range(T_EVAL):
                    # Collect K×H embeddings for trajectory t (same order as gradient computation)
                    traj_prefixes = []
                    for q in range(K):
                        prompt = eval_samples[q][t][0]
                        prefixes = build_word_prefixes(prompt, H)
                        traj_prefixes.extend(prefixes)

                    traj_embeddings = embed_texts_batched(traj_prefixes, embedder, batch_size=128)
                    # Shape: (K × H, d)

                    # Compute Fisher same as gradient: T × data + λI
                    fisher_t = compute_fisher_from_embeddings(
                        traj_embeddings, k=K, h=H, lambda_reg=LAMBDA_REG, t_coef=T
                    )
                    sign, logdet = torch.linalg.slogdet(fisher_t)
                    if sign > 0:
                        eval_logdets.append(logdet.item())

                # Average logdet over T_EVAL samples (same as gradient averaging over M)
                eval_obj = sum(eval_logdets) / len(eval_logdets) if eval_logdets else float('-inf')
                eval_obj_str = f"{eval_obj:12.4f}"

        # Select prefix for this iteration (random if using prefix_list)
        if prefix_list:
            current_prefix = random.choice(prefix_list)
        else:
            current_prefix = PROMPT_PREFIX

        # Step 1: Generate M×K prompts in parallel (batched generation per policy)
        def gen_m_samples_batched(policy_idx, prefix=current_prefix):
            policy = policy_manager.policies[policy_idx]
            return policy.generate_until_h_words_batched(
                batch_size=M, h_words=H, temperature=TEMPERATURE, prompt_prefix=prefix
            )

        with ThreadPoolExecutor(max_workers=K) as executor:
            all_policy_samples = list(executor.map(gen_m_samples_batched, range(K)))
        # all_policy_samples[q][m] = (prompt, logprobs, boundaries) for policy q, sample m

        # Step 2: Collect prompts/prefixes for embedding
        sample_prompts = [all_policy_samples[q][0][0] for q in range(K)]  # First sample for logging

        if NO_INTERMEDIATE:
            # Only embed final prompts: M×K total
            all_texts_flat = []
            for m in range(M):
                for q in range(K):
                    prompt = all_policy_samples[q][m][0]
                    all_texts_flat.append(prompt)
            all_embeddings = embed_texts_batched(all_texts_flat, embedder, batch_size=128)
            # Shape: (M×K, d)
            H_eff = 1  # Effective H for Fisher computation
        else:
            # Embed all prefixes: M×K×H total
            all_prefixes_flat = []
            for m in range(M):
                for q in range(K):
                    prompt = all_policy_samples[q][m][0]
                    prefixes = build_word_prefixes(prompt, H)
                    all_prefixes_flat.extend(prefixes)
            all_embeddings = embed_texts_batched(all_prefixes_flat, embedder, batch_size=128)
            # Shape: (M×K×H, d)
            H_eff = H

        # Step 4: Compute L_m for each sample and accumulate weighted gradients
        L_values = []
        policy_params = list(policy_manager.policies[policy_to_update].get_trainable_parameters())
        grad_accum = [torch.zeros_like(p) for p in policy_params]

        for m_idx in range(M):
            # Extract embeddings for sample m: K×H_eff embeddings
            start_idx = m_idx * K * H_eff
            end_idx = start_idx + K * H_eff
            embeddings_m = all_embeddings[start_idx:end_idx]

            logprobs_m = all_policy_samples[policy_to_update][m_idx][1]
            word_boundaries = all_policy_samples[policy_to_update][m_idx][2]
            policy_device = policy_manager.devices[policy_to_update]

            # Compute Fisher and objective (shared across all baseline types except per-word)
            fisher_m = compute_fisher_from_embeddings(embeddings_m, K, H_eff, lambda_reg=LAMBDA_REG, t_coef=T)

            # Compute objective based on design
            if DESIGN == "D":
                sign, logdet = torch.linalg.slogdet(fisher_m)
                if sign <= 0:
                    print(f"Warning: Fisher not positive definite at iter {iteration}, sample {m_idx}")
                    continue
                L_m = logdet
            elif DESIGN == "A":
                try:
                    fisher_inv = torch.linalg.inv(fisher_m)
                    L_m = -torch.trace(fisher_inv)
                except RuntimeError:
                    print(f"Warning: Fisher not invertible at iter {iteration}, sample {m_idx}")
                    continue
            else:  # V-optimal
                try:
                    V_m = compute_V_matrix(embeddings_m, K, H_eff)
                    fisher_inv = torch.linalg.inv(fisher_m)
                    L_m = -torch.trace(V_m @ fisher_inv)
                except RuntimeError:
                    print(f"Warning: Fisher not invertible at iter {iteration}, sample {m_idx}")
                    continue

            L_values.append(L_m.item())

            if BASELINE == "per-word" and not NO_INTERMEDIATE:
                # Per-word baseline: each word w gets its own objective L_w
                L_w_list, _ = compute_per_word_objectives(
                    embeddings_m, K, H_eff, lambda_reg=LAMBDA_REG, t_coef=T, design=DESIGN
                )

                # Compute gradient: Σ_w L_w * ∇log π(word_w)
                prev_boundary = 0
                for w in range(min(len(word_boundaries), H_eff)):
                    boundary = word_boundaries[w]
                    L_w = L_w_list[w]

                    if L_w == float('-inf'):
                        prev_boundary = boundary
                        continue

                    word_logprob = 0
                    for token_idx in range(prev_boundary, boundary):
                        if token_idx < len(logprobs_m):
                            word_logprob = word_logprob + logprobs_m[token_idx]

                    if word_logprob != 0:
                        grads_w = torch.autograd.grad(
                            word_logprob, policy_params,
                            retain_graph=(m_idx < M - 1 or w < len(word_boundaries) - 1)
                        )
                        for i, g in enumerate(grads_w):
                            grad_accum[i] = grad_accum[i] + g * L_w

                    prev_boundary = boundary

            elif BASELINE == "weighted" and not NO_INTERMEDIATE:
                # Weighted approach: weight by how many embeddings each word affects
                weighted_logprob = 0
                prev_boundary = 0
                for w in range(len(word_boundaries)):
                    boundary = word_boundaries[w]
                    weight = H_eff - w  # word w affects (H_eff - w) embeddings
                    for token_idx in range(prev_boundary, boundary):
                        if token_idx < len(logprobs_m):
                            weighted_logprob = weighted_logprob + logprobs_m[token_idx] * weight
                    prev_boundary = boundary

                total_weight = H_eff * (H_eff + 1) / 2
                grads_m = torch.autograd.grad(weighted_logprob, policy_params, retain_graph=(m_idx < M - 1))
                L_m_scaled = L_m.detach().to(policy_device) / total_weight
                for i, g in enumerate(grads_m):
                    grad_accum[i] = grad_accum[i] + g * L_m_scaled

            else:  # BASELINE == "none" or NO_INTERMEDIATE
                # Simple: L * sum(logprobs) - no weighting
                total_logprob = sum(logprobs_m)
                grads_m = torch.autograd.grad(total_logprob, policy_params, retain_graph=(m_idx < M - 1))
                L_m_val = L_m.detach().to(policy_device)
                for i, g in enumerate(grads_m):
                    grad_accum[i] = grad_accum[i] + g * L_m_val

        if len(L_values) == 0:
            print(f"Warning: All Fishers singular at iter {iteration}")
            continue

        # Average gradient over M samples
        optimizers[policy_to_update].zero_grad()
        for param, g_acc in zip(policy_params, grad_accum):
            param.grad = -g_acc / M  # Negative for ascent, average over M
        optimizers[policy_to_update].step()
        if schedulers is not None:
            schedulers[policy_to_update].step()

        L = sum(L_values) / len(L_values)  # Average objective for logging

        iter_time = time.perf_counter() - iter_start

        # Log every iteration
        if True:
            clean_prompt = sample_prompts[0].replace("<|endoftext|>", "").strip()
            prompt_preview = clean_prompt[:40] + "..." if len(clean_prompt) > 40 else clean_prompt
            print(f"{iteration:>5} | {L:>12.4f} | {eval_obj_str:>12} | {iter_time:>7.2f}s | {prompt_preview}")

        # Print all K prompts at iteration 0
        if iteration == 0:
            print("\n  Initial K prompts:")
            for q, prompt in enumerate(sample_prompts):
                print(f"    Policy {q}: {prompt[:60]}...")
            print()

    print("=" * 60)
    print("Final prompts (no prefix):")
    for q, policy in enumerate(policy_manager.policies):
        results = policy.generate_until_h_words_batched(batch_size=1, h_words=H, temperature=1.0, prompt_prefix="")
        clean_prompt = results[0][0].replace("<|endoftext|>", "").strip()
        print(f"    Policy {q}: {clean_prompt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
