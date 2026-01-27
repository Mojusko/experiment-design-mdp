#!/usr/bin/env python3
"""
Sample prompts from optimized policies saved by reinforce_word_level.py.

Usage:
    python sample_saved_policies.py --checkpoint results/.../policies.pt
"""

import argparse
import os
import sys
from typing import Any, Dict

import torch

# Allow importing reinforce_word_level.py as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reinforce_word_level import MultiGPUPolicyManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample prompts from saved word-level policies")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policies.pt checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Override model name (if not stored in checkpoint)")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon H (words per prompt)")
    parser.add_argument("--num-samples-per-policy", type=int, default=5, help="Samples to draw per policy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--prefix", type=str, default="", help="Optional prompt prefix")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU. By default, policies are placed across available GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Force CPU by hiding GPUs before any CUDA initialization
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")

    saved_args: Dict[str, Any] = checkpoint.get("args", {})
    policies_state = checkpoint.get("policies")
    if not policies_state:
        raise ValueError("Checkpoint does not contain 'policies'")

    model_name = args.model or checkpoint.get("model_name") or saved_args.get("model")
    if not model_name:
        raise ValueError("Could not determine model name. Pass --model explicitly.")

    horizon = args.horizon or checkpoint.get("horizon") or saved_args.get("horizon")
    if horizon is None:
        raise ValueError("Could not determine horizon. Pass --horizon explicitly.")

    num_policies = len(policies_state)
    base_seed = int(saved_args.get("seed", 42))

    device_label = "cpu" if args.cpu else "cuda"
    print(f"Rebuilding {num_policies} policies for model={model_name} on device={device_label}")
    policy_manager = MultiGPUPolicyManager(
        k=num_policies,
        model_name=model_name,
        base_seed=base_seed,
        init_noise=0.0,
    )

    # Load optimized weights
    for q, state_dict in enumerate(policies_state):
        policy = policy_manager.policies[q]
        policy.model.load_state_dict(state_dict)

    print("=" * 60)
    print("Sampling prompts from optimized policies")
    print("=" * 60)

    for q, policy in enumerate(policy_manager.policies):
        results = policy.generate_until_h_words_batched(
            batch_size=args.num_samples_per_policy,
            h_words=horizon,
            temperature=args.temperature,
            prompt_prefix=args.prefix,
        )
        print(f"\nPolicy {q}:")
        for i, (prompt, _, _) in enumerate(results):
            clean_prompt = prompt.replace("<|endoftext|>", "").strip()
            print(f"  {i:>2}: {clean_prompt}")


if __name__ == "__main__":
    main()
