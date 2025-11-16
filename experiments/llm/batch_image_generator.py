#!/usr/bin/env python
"""Batch image generation script for benchmarking."""
import argparse
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
import PIL.Image

from image_generator import StableDiffusionGenerator, DEFAULT_CONFIG
from components.embedder import create_embedder


def main():
    parser = argparse.ArgumentParser(description="Generate multiple images in batch")
    parser.add_argument("--prompts", nargs='+', required=True,
                       help="Space-separated list of prompts")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (if None, uses len(prompts))")
    parser.add_argument("--num_inference_steps", type=int,
                       default=DEFAULT_CONFIG["num_inference_steps"],
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float,
                       default=DEFAULT_CONFIG["guidance_scale"],
                       help="Guidance scale")
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"],
                       help="Image size")
    parser.add_argument("--embedder_model_id", type=str,
                       default="openai/clip-vit-large-patch14",
                       help="Embedder model ID")
    parser.add_argument("--sequential", action='store_true',
                       help="Generate images sequentially (for comparison)")

    args = parser.parse_args()

    # Setup embedder
    embedder_cfg = OmegaConf.create({
        "_target_": "components.embedder.CLIPEmbedder",
        "model_id": args.embedder_model_id,
        "normalize": True,
        "cache_dir": DEFAULT_CONFIG["MODELS_CACHE_DIR"]
    })
    embedder = create_embedder(embedder_cfg)
    print(f"Initialized Embedder: {embedder.__class__.__name__}")

    # Initialize generator
    generator = StableDiffusionGenerator(
        stable_diffusion_id=DEFAULT_CONFIG["stable_diffusion_id"],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    num_prompts = len(args.prompts)
    print(f"\nGenerating {num_prompts} images...")
    print(f"Prompts: {args.prompts}")
    print(f"Mode: {'Sequential' if args.sequential else 'Batched'}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}\n")

    start_time = time.time()

    if args.sequential:
        # Sequential generation (baseline)
        for i, prompt in enumerate(args.prompts):
            print(f"Generating image {i+1}/{num_prompts}: {prompt[:50]}...")
            img_start = time.time()
            image_np, image_embedding = generator.sample(prompt, embedder)
            img_time = time.time() - img_start

            # Save image
            filename = f"seq_{i:03d}.png"
            filepath = os.path.join(args.output_dir, filename)
            PIL.Image.fromarray(image_np).save(filepath)
            print(f"  Saved to {filename} ({img_time:.2f}s)")
    else:
        # Batch generation
        batch_size = args.batch_size if args.batch_size else num_prompts
        print(f"Batch size: {batch_size}")

        for batch_idx in range(0, num_prompts, batch_size):
            batch_prompts = args.prompts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            print(f"\nBatch {batch_num}: Generating {len(batch_prompts)} images...")

            batch_start = time.time()
            results = generator.sample_batch(batch_prompts, embedder)
            batch_time = time.time() - batch_start

            print(f"  Batch completed in {batch_time:.2f}s ({batch_time/len(batch_prompts):.2f}s per image)")

            # Save images
            for i, (image_np, image_embedding) in enumerate(results):
                img_idx = batch_idx + i
                filename = f"batch_{img_idx:03d}.png"
                filepath = os.path.join(args.output_dir, filename)
                PIL.Image.fromarray(image_np).save(filepath)
                print(f"  Saved {filename}")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total images: {num_prompts}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per image: {total_time/num_prompts:.2f}s")
    print(f"  Mode: {'Sequential' if args.sequential else f'Batched (batch_size={batch_size})'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
