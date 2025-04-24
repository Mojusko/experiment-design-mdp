import logging
import argparse
import os
import PIL.Image
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer # Keep these for SD text encoding
# Removed CLIPModel, CLIPProcessor imports for image embedding here
# Import BaseEmbedder for type hinting
from components.embedder import BaseEmbedder, create_embedder # Added create_embedder for main block
# Define PILImage type for type hinting
PILImage = PIL.Image.Image

import hashlib

logger = logging.getLogger(__name__)

def _get_seed_from_prompt(prompt: str) -> int:
    # Compute SHA-256 hash of the prompt and convert to an integer.
    hash_digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    # Convert the hex digest to an integer and constrain it to 32 bits
    return int(hash_digest, 16) % (2**32)

# Default configuration for image generation
DEFAULT_CONFIG = {
    "stable_diffusion_id": "CompVis/stable-diffusion-v1-4",
    "num_inference_steps": 100,
    "guidance_scale": 8.0, # Renamed from guidance_base
    "image_size": 512,
    "seed": 0, # Default base seed
    "MODELS_CACHE_DIR": os.path.expanduser("~/.cache/huggingface/hub"),
    "output_dir": "generated_images"
}

class StableDiffusionGenerator():
    def __init__(
        self,
        # Make stable_diffusion_id a keyword argument with default from DEFAULT_CONFIG
        stable_diffusion_id: str = DEFAULT_CONFIG['stable_diffusion_id'],
        num_inference_steps: int = DEFAULT_CONFIG['num_inference_steps'],
        guidance_scale: float = DEFAULT_CONFIG['guidance_scale'],
        image_size: int = DEFAULT_CONFIG['image_size'],
        seed: int = DEFAULT_CONFIG['seed'],
        MODELS_CACHE_DIR: str = DEFAULT_CONFIG['MODELS_CACHE_DIR']
    ) -> None:
        """An implementation of stable diffusion's text2image generator.

        Args:
            stable_diffusion_id (str): The stable diffusion's model identifier. Defaults to DEFAULT_CONFIG.
            num_inference_steps (int): The number of denoising steps. Defaults to DEFAULT_CONFIG.
            guidance_scale (float): The guidance scale for classifier-free guidance. Defaults to DEFAULT_CONFIG.
            image_size (int): Size of generated images.
            seed (int): Random seed for reproducibility. Defaults to DEFAULT_CONFIG.
            MODELS_CACHE_DIR (str): Directory to cache the downloaded models. Defaults to DEFAULT_CONFIG.
        """
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.MODELS_CACHE_DIR = MODELS_CACHE_DIR # Store cache dir
        self._image_size = image_size # Store image size

        # Remove internal CLIP loading for image embedding
        # self._clip_model = ...
        # self._clip_processor = ...

        # Load tokenizer and text encoder *specifically for Stable Diffusion's text conditioning*
        # This is separate from the embedder used for scoring/analysis.
        self._tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_id,
            subfolder="tokenizer",
            cache_dir=str(MODELS_CACHE_DIR)
        )
        
        self._text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_id,
            subfolder="text_encoder",
            cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        
        # Load UNet for denoising
        self._unet = UNet2DConditionModel.from_pretrained(
            stable_diffusion_id,
            subfolder="unet",
            cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        
        # Load VAE for image encoding/decoding
        self._vae = AutoencoderKL.from_pretrained(
            stable_diffusion_id,
            subfolder="vae",
            cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        
        # Setup noise scheduler
        self._scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )
        
        self.seed_generator()
        self.latents = None

    def seed_generator(self) -> None:
        # Setup random generator for SD noise
        if self.seed:
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.seed()

    @torch.no_grad()
    def resample_random(self) -> None:
        """Generates new random latents for image generation."""
        latents_height = self._image_size // 8
        latents_width = self._image_size // 8
        # Access in_channels via config to avoid FutureWarning
        in_channels = self._unet.config.in_channels
        self.latents = torch.randn(
            (1, in_channels, latents_height, latents_width),
            generator=self._generator,
            device=self.device
        )

    @torch.no_grad()
    def sample(self, prompt: str, embedder: BaseEmbedder, raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """Generates an image from a text prompt and embeds it using the provided embedder.

        Args:
            prompt (str): The text prompt to generate an image from.
            embedder (BaseEmbedder): The embedder instance to use for image embedding.
            raw (bool): If True, returns raw image tensor alongside processed numpy array.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
                - If raw=False: (numpy image array [H, W, C], image embedding tensor [1, D])
                - If raw=True: (numpy image array [H, W, C], raw image tensor, image embedding tensor [1, D])
        """
        # Set SD seed based on the prompt for deterministic generation
        self.seed = _get_seed_from_prompt(prompt)
        self.seed_generator()

        if self.latents is None:
            self.resample_random()

        # Encode the prompt
        text_input = self._tokenizer(
            [prompt],
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Get text embeddings
        text_embeddings = self._text_encoder(text_input.input_ids.to(self.device))[0]

        # Create unconditioned embeddings for classifier-free guidance
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self._tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self._text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latents
        latents = self.latents.to(self.device)
        self._scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self._scheduler.init_noise_sigma

        # Denoising loop
        for t in self._scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self._scheduler.scale_model_input(latent_model_input, timestep=t)

            # Predict noise residual
            noise_pred = self._unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image
        latents = 1 / 0.18215 * latents
        image = self._vae.decode(latents).sample
        
        image_raw = image.clone()

        # Process image for output
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0]

        # Convert to PIL Image and get embedding using the provided embedder
        pil_image = PIL.Image.fromarray(image)
        # Ensure embedder is on the same device potentially? Or handle internally.
        # Assuming embedder handles device placement.
        image_embedding = embedder.embed_image(pil_image) # Use the passed embedder
        # Keep embedding as a tensor [1, D] on its original device

        if raw:
            # Return numpy image, raw tensor, embedding tensor
            return image, image_raw.detach().cpu(), image_embedding.detach()
        # Return numpy image, embedding tensor
        return image, image_embedding.detach()

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Returns the output image dimensions.

        Returns:
            Tuple[int, int, int]: (height, width, channels)
        """
        return (self._image_size, self._image_size, 3)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate an image using StableDiffusionGenerator")
    parser.add_argument("--base_prompt", type=str, required=True, help="Base prompt for image generation (e.g., 'A man walking in paris')")
    parser.add_argument("--full_prompt", type=str, required=True, help="Full prompt for image generation (e.g., 'A man walking in paris #photorealistic #cute')")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help=f"Directory to save the generated image (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG['seed']})")
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"], help=f"Size of the generated image (default: {DEFAULT_CONFIG['image_size']})")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_CONFIG["num_inference_steps"], help=f"Number of inference steps (default: {DEFAULT_CONFIG['num_inference_steps']})")
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_CONFIG["guidance_scale"], help=f"Guidance scale for prompt (default: {DEFAULT_CONFIG['guidance_scale']})") # Updated arg name
    # Removed guidance_tokens argument as we are using the single-guidance generator for now
    # parser.add_argument("--guidance_tokens", type=float, default=DEFAULT_CONFIG["guidance_tokens"], help=f"Guidance scale for full prompt tokens (default: {DEFAULT_CONFIG['guidance_tokens']})")
    parser.add_argument("--embedder_model_id", type=str, default="openai/clip-vit-large-patch14", help="Model ID for the embedder (e.g., CLIP or SigLIP)")
    parser.add_argument("--embedder_normalize", type=bool, default=True, help="Whether the embedder should normalize features")


    args = parser.parse_args()

    # --- Setup Embedder ---
    # Create a dummy config for the embedder based on args
    from omegaconf import OmegaConf
    embedder_cfg = OmegaConf.create({
        # Assuming CLIPEmbedder for now, adjust if needed or make configurable
        "_target_": "experiments.llm.components.embedder.CLIPEmbedder",
        "model_id": args.embedder_model_id,
        "normalize": args.embedder_normalize,
        "cache_dir": DEFAULT_CONFIG["MODELS_CACHE_DIR"]
    })
    embedder = create_embedder(embedder_cfg)
    print(f"Initialized Embedder: {embedder.__class__.__name__} with model {embedder.model_id}")


    # --- Print Configuration ---
    print("Using configuration:")
    print(f"  base_prompt: '{args.base_prompt}'")
    print(f"  full_prompt: '{args.full_prompt}'")
    print(f"  stable_diffusion_id: {DEFAULT_CONFIG['stable_diffusion_id']}")
    print(f"  num_inference_steps: {args.num_inference_steps}")
    print(f"  guidance_scale: {args.guidance_scale}") # Updated arg name
    print(f"  image_size: {args.image_size}")
    print(f"  seed: {args.seed}") # Note: Seed is now derived from prompt internally
    print(f"  output_dir: {args.output_dir}")
    print(f"  embedder_model_id: {args.embedder_model_id}")
    print(f"  embedder_normalize: {args.embedder_normalize}")

    # Initialize the SD generator (using single guidance version for simplicity here)
    generator = StableDiffusionGenerator(
        stable_diffusion_id=DEFAULT_CONFIG["stable_diffusion_id"],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, # Updated arg name
        image_size=args.image_size,
        # Seed is derived from prompt internally
        MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"]
    )

    # Generate the image using the full prompt and the created embedder
    # The base_prompt arg is ignored by StableDiffusionGenerator.sample
    # We get back the numpy image and the embedding tensor
    image_np, image_embedding = generator.sample(args.full_prompt, embedder=embedder)

    # NOTE: This example uses StableDiffusionGenerator. If using DoubleGuidanceStableDiffusionGenerator,
    # you would call:
    # image_np, image_embedding = generator.sample(args.base_prompt, args.full_prompt, embedder=embedder)

    print(f"Generated image embedding shape: {image_embedding.shape}, dtype: {image_embedding.dtype}, device: {image_embedding.device}")

    # Create the output directory if it doesn’t exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a filename that includes prompt and parameters
    # Sanitize the prompt for filename use
    sanitized_prompt = args.full_prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized_prompt = ''.join(c for c in sanitized_prompt if c.isalnum() or c in '_-#')[:50] # Limit length

    # Update filename to reflect embedder used (optional)
    embedder_name_short = embedder.__class__.__name__.replace("Embedder","").lower()
    filename = f"{sanitized_prompt}_guidance{args.guidance_scale}_steps{args.num_inference_steps}_emb-{embedder_name_short}.png" # Updated arg name

    # Save the image with the descriptive filename
    image_path = os.path.join(args.output_dir, filename)
    PILImage.fromarray(image_np).save(image_path) # Save the numpy image
    print(f"Image saved to {image_path}")

    # Optionally save the embedding too
    embedding_path = os.path.splitext(image_path)[0] + ".pt"
    torch.save(image_embedding, embedding_path)
    print(f"Image embedding saved to {embedding_path}")
