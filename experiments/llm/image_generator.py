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
# Removed unused PILImage type hint alias

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


class TripleGuidanceStableDiffusionGenerator():
    """
    Stable Diffusion generator implementing triple guidance:
    1. Unconditional
    2. Text Prompt
    3. Estimator Vector (Preference Direction)
    """
    def __init__(
        self,
        stable_diffusion_id: str = DEFAULT_CONFIG['stable_diffusion_id'],
        num_inference_steps: int = DEFAULT_CONFIG['num_inference_steps'],
        guidance_scale: float = DEFAULT_CONFIG['guidance_scale'], # Scale for text prompt
        guidance_scale_2: float = 1.0, # Scale for estimator vector (needs tuning)
        image_size: int = DEFAULT_CONFIG['image_size'],
        seed: int = DEFAULT_CONFIG['seed'], # Base seed, but prompt seed overrides in sample
        MODELS_CACHE_DIR: str = DEFAULT_CONFIG['MODELS_CACHE_DIR'],
        normalize_estimator: bool = True # Flag to normalize estimator embedding
    ) -> None:
        """
        Args:
            stable_diffusion_id (str): SD model identifier.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for the text prompt condition.
            guidance_scale_2 (float): Guidance scale for the estimator vector condition.
            image_size (int): Size of generated images.
            seed (int): Base random seed (overridden by prompt-specific seed in sample).
            MODELS_CACHE_DIR (str): Directory for model cache.
            normalize_estimator (bool): Whether to L2 normalize the estimator embedding before use.
        """
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale # Text prompt scale
        self.guidance_scale_2 = guidance_scale_2 # Estimator scale
        self.normalize_estimator = normalize_estimator # Store normalization flag
        self.MODELS_CACHE_DIR = MODELS_CACHE_DIR
        self._image_size = image_size

        # Load SD components (tokenizer, text encoder, unet, vae, scheduler)
        # This is identical to StableDiffusionGenerator initialization
        self._tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_id, subfolder="tokenizer", cache_dir=str(MODELS_CACHE_DIR)
        )
        self._text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_id, subfolder="text_encoder", cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        self._unet = UNet2DConditionModel.from_pretrained(
            stable_diffusion_id, subfolder="unet", cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        self._vae = AutoencoderKL.from_pretrained(
            stable_diffusion_id, subfolder="vae", cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        self._scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

        self.seed_generator() # Initialize base generator state
        self.latents = None # Initialize latents

    def seed_generator(self) -> None:
        """Sets up the random generator based on the current seed."""
        if self.seed is not None: # Allow seed to be None for non-deterministic base
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.seed() # Seed with system randomness

    @torch.no_grad()
    def resample_random(self) -> None:
        """Generates new random latents for image generation."""
        latents_height = self._image_size // 8
        latents_width = self._image_size // 8
        in_channels = self._unet.config.in_channels
        # Use the generator initialized/seeded by seed_generator
        self.latents = torch.randn(
            (1, in_channels, latents_height, latents_width),
            generator=self._generator,
            device=self.device
        )

    @torch.no_grad()
    def sample(self, prompt: str, estimator_embedding: torch.Tensor, embedder: BaseEmbedder, raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """
        Generates an image using triple guidance (uncond, prompt, estimator).

        Args:
            prompt (str): The text prompt.
            estimator_embedding (torch.Tensor): The learned estimator vector [1, D].
            embedder (BaseEmbedder): Embedder instance for final image embedding.
            raw (bool): If True, return raw image tensor.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
                - If raw=False: (numpy image array [H, W, C], image embedding tensor [1, D])
                - If raw=True: (numpy image array [H, W, C], raw image tensor, image embedding tensor [1, D])
        """
        # --- Seeding ---
        # Set SD seed based on the prompt for deterministic generation per prompt
        self.seed = _get_seed_from_prompt(prompt)
        self.seed_generator() # Re-seed the generator for this specific prompt

        # --- Initial Latents ---
        # Generate initial noise using the prompt-seeded generator
        self.resample_random() # Generate noise based on the current self._generator state
        latents = self.latents.to(self.device) # Use the generated noise

        # --- Prepare Embeddings ---
        # 1. Unconditional Embedding
        max_length = self._tokenizer.model_max_length
        uncond_input = self._tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self._text_encoder(uncond_input.input_ids.to(self.device))[0] # Shape: [1, 77, 768]

        # 2. Text Prompt Embedding
        text_input = self._tokenizer(
            [prompt], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = self._text_encoder(text_input.input_ids.to(self.device))[0] # Shape: [1, 77, 768]

        # 3. Estimator Embedding
        # Ensure estimator embedding is on the correct device and has the right shape
        estimator_emb = estimator_embedding.to(self.device).to(self._text_encoder.dtype) # Match dtype
        if estimator_emb.shape[0] != 1:
             estimator_emb = estimator_emb.unsqueeze(0) # Ensure shape [1, D]

        # Check if estimator embedding dimension matches text embedding dimension
        # Note: SD uses pooled output [1, 768], estimator might be [1, 512] or similar.
        # We need to project or adapt the estimator embedding to match the expected input dim [1, 77, 768]
        # For now, we'll replicate the [1, D] embedding across the sequence length (77)
        # This is a simplification; more sophisticated projection might be needed.
        target_dim = uncond_embeddings.shape[-1] # e.g., 768
        if estimator_emb.shape[-1] != target_dim:
             # Simple projection if dimensions mismatch (e.g., linear layer) - Placeholder!
             # raise ValueError(f"Estimator embedding dim ({estimator_emb.shape[-1]}) must match text embedding dim ({target_dim}). Projection needed.")
             # For now, let's try repeating. This might not be ideal.
             print(f"Warning: Estimator embedding dim ({estimator_emb.shape[-1]}) differs from text embedding dim ({target_dim}). Repeating embedding across sequence length.")
             if estimator_emb.shape[-1] > target_dim:
                 estimator_emb = estimator_emb[:, :target_dim] # Truncate if larger
             elif estimator_emb.shape[-1] < target_dim:
                 padding = torch.zeros(1, target_dim - estimator_emb.shape[-1], device=self.device, dtype=estimator_emb.dtype)
                 estimator_emb = torch.cat([estimator_emb, padding], dim=1) # Pad if smaller

        # Replicate across sequence length dimension (77)
        estimator_embeddings_seq = estimator_emb.unsqueeze(1).repeat(1, max_length, 1) # Shape: [1, 77, 768]

        # Normalize estimator embedding if requested
        if self.normalize_estimator:
            estimator_embeddings_seq = torch.nn.functional.normalize(estimator_embeddings_seq, p=2, dim=-1)

        # Concatenate all three embeddings for UNet input
        # Order: Unconditional, Text, Estimator
        combined_embeddings = torch.cat([uncond_embeddings, text_embeddings, estimator_embeddings_seq])

        # --- Denoising Loop ---
        self._scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self._scheduler.init_noise_sigma # Scale initial noise

        for t in self._scheduler.timesteps:
            # Expand latents for the three conditions
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = self._scheduler.scale_model_input(latent_model_input, timestep=t)

            # Predict noise residual for all conditions
            noise_pred = self._unet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings
            ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text, noise_pred_estimator = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + \
                         self.guidance_scale * (noise_pred_text - noise_pred_uncond) + \
                         self.guidance_scale_2 * (noise_pred_estimator - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        # --- Decode and Post-process ---
        latents = 1 / 0.18215 * latents
        image = self._vae.decode(latents).sample
        image_raw = image.clone() # Keep raw tensor if needed

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0] # Final numpy image [H, W, C]

        # --- Embed Final Image ---
        pil_image = PIL.Image.fromarray(image)
        image_embedding = embedder.embed_image(pil_image) # Use the passed embedder

        if raw:
            return image, image_raw.detach().cpu(), image_embedding.detach()
        return image, image_embedding.detach()

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Returns the output image dimensions."""
        return (self._image_size, self._image_size, 3)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate an image using StableDiffusionGenerator")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation (e.g., 'A cat, fluffy, sitting on a mat')")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help=f"Directory to save the generated image (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG['seed']})") # Note: Seed is derived from prompt internally by generators
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"], help=f"Size of the generated image (default: {DEFAULT_CONFIG['image_size']})")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_CONFIG["num_inference_steps"], help=f"Number of inference steps (default: {DEFAULT_CONFIG['num_inference_steps']})")
    parser.add_argument("--guidance_prompt", type=float, default=DEFAULT_CONFIG["guidance_scale"], help=f"Guidance scale for text prompt (default: {DEFAULT_CONFIG['guidance_scale']})")
    parser.add_argument("--guidance_estimator", type=float, default=1.0, help="Guidance scale for estimator vector (default: 1.0)")
    parser.add_argument("--estimator", type=str, default=None, help="Optional estimator: path to .pt file or text to embed for guidance")
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
    print(f"  prompt: '{args.prompt}'")
    print(f"  stable_diffusion_id: {DEFAULT_CONFIG['stable_diffusion_id']}")
    print(f"  num_inference_steps: {args.num_inference_steps}")
    print(f"  guidance_prompt: {args.guidance_prompt}")
    print(f"  guidance_estimator: {args.guidance_estimator}")
    print(f"  estimator: {args.estimator}")
    print(f"  image_size: {args.image_size}")
    print(f"  seed: {args.seed}") # Note: Seed is derived from prompt internally by generators
    print(f"  output_dir: {args.output_dir}")
    print(f"  embedder_model_id: {args.embedder_model_id}")
    print(f"  embedder_normalize: {args.embedder_normalize}")

    # --- Initialize Generator and Generate Image ---
    estimator_embedding = None
    generator_type = "standard"

    if args.estimator:
        print(f"Estimator provided: '{args.estimator}'")
        # Check if it's a path using standard os functions
        estimator_path = os.path.abspath(args.estimator) # Use os.path.abspath
        if ('/' in args.estimator or '\\' in args.estimator) and os.path.exists(estimator_path):
            print(f"Loading estimator tensor from path: {estimator_path}")
            try:
                estimator_embedding = torch.load(estimator_path, map_location=embedder.device)
                # Ensure it's a tensor and has the correct shape (e.g., [1, D] or [D])
                if not isinstance(estimator_embedding, torch.Tensor):
                     raise TypeError(f"Loaded estimator is not a tensor (type: {type(estimator_embedding)})")
                if estimator_embedding.dim() == 1:
                     estimator_embedding = estimator_embedding.unsqueeze(0) # Ensure [1, D]
                print(f"Loaded estimator tensor with shape: {estimator_embedding.shape}")
                generator_type = "triple_guidance_path"
            except Exception as e:
                print(f"Error loading estimator from path '{estimator_path}': {e}. Proceeding without estimator guidance.")
                estimator_embedding = None # Reset on error
        else:
            print(f"Treating estimator as text: '{args.estimator}'")
            try:
                estimator_embedding = embedder.embed_text(args.estimator)
                print(f"Embedded estimator text to shape: {estimator_embedding.shape}")
                generator_type = "triple_guidance_text"
            except Exception as e:
                print(f"Error embedding estimator text '{args.estimator}': {e}. Proceeding without estimator guidance.")
                estimator_embedding = None # Reset on error

    # Instantiate the appropriate generator
    if estimator_embedding is not None:
        print("Using Triple Guidance Generator")
        generator = TripleGuidanceStableDiffusionGenerator(
            stable_diffusion_id=DEFAULT_CONFIG["stable_diffusion_id"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_prompt, # Text prompt scale
            guidance_scale_2=args.guidance_estimator, # Estimator scale
            image_size=args.image_size,
            # Seed is derived from prompt internally
            MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"]
            # normalize_estimator=True # Keep default or make configurable? Default is True
        )
        # Generate image using triple guidance
        image_np, image_embedding = generator.sample(args.prompt, estimator_embedding, embedder=embedder)
    else:
        print("Using Standard Stable Diffusion Generator")
        generator = StableDiffusionGenerator(
            stable_diffusion_id=DEFAULT_CONFIG["stable_diffusion_id"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_prompt, # Standard guidance scale
            image_size=args.image_size,
            # Seed is derived from prompt internally
            MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"]
        )
        # Generate image using standard guidance
        image_np, image_embedding = generator.sample(args.prompt, embedder=embedder)

    print(f"Generated image embedding shape: {image_embedding.shape}, dtype: {image_embedding.dtype}, device: {image_embedding.device}")

    # Create the output directory if it doesn’t exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a filename that includes prompt and parameters
    # Sanitize the prompt for filename use
    sanitized_prompt = args.prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized_prompt = ''.join(c for c in sanitized_prompt if c.isalnum() or c in '_-#')[:50] # Limit length

    # Determine estimator string for filename
    estimator_string = "no_est" # Default if no estimator is used
    if generator_type == "triple_guidance_text":
        # Sanitize the estimator text itself
        sanitized_estimator_text = args.estimator.replace(' ', '_').replace('/', '_').replace('\\', '_')
        estimator_string = ''.join(c for c in sanitized_estimator_text if c.isalnum() or c in '_-#')[:30] # Limit length
    elif generator_type == "triple_guidance_path":
        estimator_string = "saved_est"

    # Construct filename using the new format: {prompt}_{estimator}_gP{guidance_prompt}_gE{guidance_estimator}.png
    filename = f"{sanitized_prompt}_{estimator_string}_gP{args.guidance_prompt}_gE{args.guidance_estimator}.png"

    # Save the image with the descriptive filename
    image_path = os.path.join(args.output_dir, filename)
    # Use PIL.Image directly
    PIL.Image.fromarray(image_np).save(image_path)
    print(f"Image saved to {image_path}")

    # Removed saving of the generated image's embedding
