#!/usr/bin/env python3
"""
Load optimized policies, sample trajectories, and generate images from them.

This script is designed to work with checkpoints saved by reinforce_word_level.py.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import PIL.Image
import torch
from omegaconf import OmegaConf

# Allow importing local modules when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.embedder import create_embedder  # noqa: E402
from image_generator import DEFAULT_CONFIG, StableDiffusionGenerator  # noqa: E402
from reinforce_word_level import MultiGPUPolicyManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample T trajectories per policy from a saved checkpoint and generate images."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policies.pt checkpoint saved by reinforce_word_level.py",
    )
    parser.add_argument(
        "-T",
        "--num-trajectories",
        type=int,
        default=8,
        help="Number of trajectories (prompts) to sample per policy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (if not stored in checkpoint)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override horizon H (words per prompt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for prompt generation",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prompt prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/sampled-images-<timestamp>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for Stable Diffusion image generation",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_CONFIG["num_inference_steps"],
        help="Stable Diffusion inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_CONFIG["guidance_scale"],
        help="Stable Diffusion guidance scale",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_CONFIG["image_size"],
        help="Generated image size (pixels)",
    )
    parser.add_argument(
        "--embedder-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Model ID for the image embedder",
    )
    parser.add_argument(
        "--embedder-normalize",
        type=bool,
        default=True,
        help="Whether to normalize embedder features",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU by hiding CUDA devices before initialization",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir_arg: Optional[str]) -> str:
    if output_dir_arg:
        return output_dir_arg
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("results", f"sampled-images-{timestamp}")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Hide GPUs before any CUDA initialization if CPU mode is requested
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_dir = resolve_output_dir(args.output_dir)
    images_dir = os.path.join(output_dir, "images")
    prompts_path = os.path.join(output_dir, "prompts.jsonl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")

    saved_args: Dict[str, Any] = checkpoint.get("args", {})
    policies_state: List[Dict[str, torch.Tensor]] = checkpoint.get("policies")
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

    print(
        f"Rebuilding {num_policies} policies for model={model_name}, "
        f"horizon={horizon}, trajectories per policy={args.num_trajectories}"
    )
    policy_manager = MultiGPUPolicyManager(
        k=num_policies,
        model_name=model_name,
        base_seed=base_seed,
        init_noise=0.0,
    )

    # Load optimized weights
    for q, state_dict in enumerate(policies_state):
        policy_manager.policies[q].model.load_state_dict(state_dict)

    # Sample prompts from each policy
    sampled_records: List[Dict[str, Any]] = []
    print("=" * 60)
    print("Sampling prompts from optimized policies")
    print("=" * 60)
    for q, policy in enumerate(policy_manager.policies):
        results = policy.generate_until_h_words_batched(
            batch_size=args.num_trajectories,
            h_words=horizon,
            temperature=args.temperature,
            prompt_prefix=args.prefix,
        )
        print(f"\nPolicy {q}:")
        for t_idx, (prompt, _, _) in enumerate(results):
            clean_prompt = prompt.replace("<|endoftext|>", "").strip()
            print(f"  traj {t_idx:>2}: {clean_prompt[:80]}")
            sampled_records.append(
                {
                    "policy": q,
                    "trajectory": t_idx,
                    "prompt": clean_prompt,
                }
            )

    # Persist prompts immediately
    with open(prompts_path, "w") as f_prompts:
        for record in sampled_records:
            f_prompts.write(json.dumps(record) + "\n")
    print(f"\nSaved prompts to: {prompts_path}")

    # Initialize embedder and image generator
    embedder_cfg = OmegaConf.create(
        {
            "_target_": "components.embedder.CLIPEmbedder",
            "model_id": args.embedder_model_id,
            "normalize": args.embedder_normalize,
            "cache_dir": DEFAULT_CONFIG["MODELS_CACHE_DIR"],
        }
    )
    embedder = create_embedder(embedder_cfg)
    print(f"Initialized embedder: {embedder.__class__.__name__} ({embedder.model_id})")

    generator = StableDiffusionGenerator(
        stable_diffusion_id=DEFAULT_CONFIG["stable_diffusion_id"],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"],
    )
    print(
        "Initialized Stable Diffusion generator "
        f"(steps={args.num_inference_steps}, guidance={args.guidance_scale}, size={args.image_size})"
    )

    # Generate images in batches
    prompts = [record["prompt"] for record in sampled_records]
    all_image_results = []
    total_prompts = len(prompts)
    print(f"\nGenerating {total_prompts} images in batches of {args.batch_size}...")
    for start_idx in range(0, total_prompts, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_prompts)
        batch_prompts = prompts[start_idx:end_idx]
        batch_num = start_idx // args.batch_size + 1
        num_batches = (total_prompts + args.batch_size - 1) // args.batch_size
        print(f"  Batch {batch_num}/{num_batches}: prompts {start_idx}..{end_idx - 1}")
        batch_results = generator.sample_batch(batch_prompts, embedder)
        all_image_results.extend(batch_results)

    # Save images and attach paths to records
    for record, (image_np, _) in zip(sampled_records, all_image_results):
        policy_idx = int(record["policy"])
        traj_idx = int(record["trajectory"])
        filename = f"policy{policy_idx:02d}_traj{traj_idx:03d}.png"
        image_path = os.path.join(images_dir, filename)
        PIL.Image.fromarray(image_np).save(image_path)
        record["image_path"] = os.path.relpath(image_path, output_dir)

    # Save metadata for later inspection/plotting
    metadata: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "model_name": model_name,
        "horizon": int(horizon),
        "num_policies": int(num_policies),
        "num_trajectories_per_policy": int(args.num_trajectories),
        "temperature": float(args.temperature),
        "prefix": args.prefix,
        "output_dir": output_dir,
        "images_dir": images_dir,
        "prompts_path": prompts_path,
        "image_config": {
            "stable_diffusion_id": DEFAULT_CONFIG["stable_diffusion_id"],
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "image_size": int(args.image_size),
            "embedder_model_id": args.embedder_model_id,
            "embedder_normalize": bool(args.embedder_normalize),
        },
        "records": sampled_records,
    }
    with open(metadata_path, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"Metadata: {metadata_path}")
    print(f"Images dir: {images_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
