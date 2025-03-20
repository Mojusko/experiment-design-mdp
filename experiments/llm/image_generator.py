import logging
import argparse
import os
from PIL import Image
from PIL import Image
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

### Original Class: StableDiffusionGenerator
class StableDiffusionGenerator():
    def __init__(
        self,
        stable_diffusion_id: str,
        num_inference_steps: int = 100, 
        guidance_scale: float = 15,
        image_size: int = 512, 
        seed: int = 0,
        MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/'
    ) -> None:
        """An implementation of stable diffusion's text2image generator.
        
        Args:
            stable_diffusion_id (str): The stable diffusion's model identifier.
            num_inference_steps (int): The number of denoising steps.
            guidance_scale (float): The guidance scale for classifier-free guidance.
            image_size (int): Size of generated images.
            seed (int): Random seed for reproducibility.
            MODELS_CACHE_DIR (str): Directory to cache the downloaded models.
        """
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Add CLIP model for image embeddings
        self._clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODELS_CACHE_DIR
        ).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODELS_CACHE_DIR
        )

        # Load tokenizer and text encoder from the SD model
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

        self._image_size = image_size
        self.latents = None

    def seed_generator(self) -> None:
        # Setup random generator
        if self.seed:
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.seed()  # Ensure random initialization even without specific seed

    @torch.no_grad()
    def resample_random(self) -> None:
        """Generates new random latents for image generation."""
        latents_height = self._image_size // 8
        latents_width = self._image_size // 8
        self.latents = torch.randn(
            (1, self._unet.in_channels, latents_height, latents_width),
            generator=self._generator,
            device=self.device
        )

    @torch.no_grad()
    def sample(self, prompt: str, raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """Generates an image from a text prompt.

        Args:
            prompt (str): The text prompt to generate an image from.
            raw (bool): If True, returns both processed and raw image tensors.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]: Generated image(s) and text embeddings
        """

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

        # Convert to PIL Image and get CLIP embedding
        pil_image = Image.fromarray(image)
        inputs = self._clip_processor(
            images=pil_image, 
            return_tensors="pt"
        ).to(self.device)
        image_embedding = self._clip_model.get_image_features(**inputs)
        image_embedding = image_embedding.detach().cpu()[0]  # Convert to numpy array

        if raw:
            return image, image_raw, image_embedding
        return image, image_embedding

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Returns the output image dimensions.

        Returns:
            Tuple[int, int, int]: (height, width, channels)
        """
        return (self._image_size, self._image_size, 3)


### New Class: DoubleGuidanceStableDiffusionGenerator
class DoubleGuidanceStableDiffusionGenerator():
    def __init__(
        self,
        stable_diffusion_id: str,
        num_inference_steps: int = 100,
        guidance_base: float = 8.0,      # Guidance for base prompt
        guidance_tokens: float = 4.0,    # Guidance for full prompt
        image_size: int = 512,
        seed: int = 0,
        MODELS_CACHE_DIR: str = os.path.expanduser("~/.cache/huggingface/hub")
    ) -> None:
        """An implementation of Stable Diffusion's text-to-image generator with separate guidance for base and full prompts.

        Args:
            stable_diffusion_id (str): The Stable Diffusion model identifier.
            num_inference_steps (int): Number of denoising steps.
            guidance_base (float): Guidance scale for the base prompt.
            guidance_tokens (float): Guidance scale for the full prompt.
            image_size (int): Size of generated images.
            seed (int): Random seed for reproducibility.
            MODELS_CACHE_DIR (str): Directory to cache downloaded models.
        """
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inference_steps = num_inference_steps
        self.guidance_base = guidance_base
        self.guidance_tokens = guidance_tokens
        self._clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODELS_CACHE_DIR
        ).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODELS_CACHE_DIR
        )
        # Load tokenizer and text encoder
        self._tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_id, subfolder="tokenizer", cache_dir=MODELS_CACHE_DIR
        )
        self._text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_id, subfolder="text_encoder", cache_dir=MODELS_CACHE_DIR
        ).to(self.device)

        # Load UNet and VAE
        self._unet = UNet2DConditionModel.from_pretrained(
            stable_diffusion_id, subfolder="unet", cache_dir=MODELS_CACHE_DIR
        ).to(self.device)
        self._vae = AutoencoderKL.from_pretrained(
            stable_diffusion_id, subfolder="vae", cache_dir=MODELS_CACHE_DIR
        ).to(self.device)

        # Setup scheduler
        self._scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )

        # Setup random generator
        self._generator = torch.Generator(device=self.device)
        if self.seed:
            self._generator.manual_seed(self.seed)
        else:
            self._generator.seed()

        self._image_size = image_size
        self.latents = None

    @torch.no_grad()
    def seed_generator(self) -> None:
        # Setup random generator
        if self.seed:
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.seed()  # Ensure random initialization even without specific seed

    @torch.no_grad()
    def resample_random(self) -> None:
        """Generates new random latents for image generation."""
        latents_height = self._image_size // 8
        latents_width = self._image_size // 8
        self.latents = torch.randn(
            (1, self._unet.in_channels, latents_height, latents_width),
            generator=self._generator, device=self.device
        )

    @torch.no_grad()
    def sample(self, base_prompt: str, full_prompt: str, raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """Generates an image using separate guidance for base prompt and full prompt.

        Args:
            base_prompt (str): The base prompt (e.g., "A man walking in paris").
            full_prompt (str): The full prompt (e.g., "A man walking in paris #photorealistic #cute").
            raw (bool): If True, returns both processed and raw image tensors.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]: Generated image(s) and text embeddings.
        """

        self.seed_generator()
        if self.latents is None:
            self.resample_random()

        # Encode unconditioned prompt
        uncond_input = self._tokenizer(
            [""], padding="max_length", max_length=self._tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeddings = self._text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Encode base prompt
        base_input = self._tokenizer(
            [base_prompt], padding="max_length", max_length=self._tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        base_embeddings = self._text_encoder(base_input.input_ids.to(self.device))[0]

        # Encode full prompt
        full_input = self._tokenizer(
            [full_prompt], padding="max_length", max_length=self._tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        full_embeddings = self._text_encoder(full_input.input_ids.to(self.device))[0]

        # Concatenate embeddings for three-way guidance
        text_embeddings = torch.cat([uncond_embeddings, base_embeddings, full_embeddings])

        # Prepare latents
        latents = self.latents.to(self.device)
        self._scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self._scheduler.init_noise_sigma

        # Denoising loop
        for t in self._scheduler.timesteps:
            # Expand latents for three predictions
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = self._scheduler.scale_model_input(latent_model_input, timestep=t)

            # Predict noise
            noise_pred = self._unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Split into three predictions
            noise_pred_uncond, noise_pred_base, noise_pred_full = noise_pred.chunk(3)

            # Combine with separate guidance scales
            noise_pred = (noise_pred_uncond +
                          self.guidance_base * (noise_pred_base - noise_pred_uncond) +
                          self.guidance_tokens * (noise_pred_full - noise_pred_base))

            # Step to previous sample
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image
        latents = latents / 0.18215
        image = self._vae.decode(latents).sample
        image_raw = image.clone()

        # Process image for output
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0]

        # Convert to PIL Image and get CLIP embedding
        pil_image = Image.fromarray(image)
        inputs = self._clip_processor(
            images=pil_image, 
            return_tensors="pt"
        ).to(self.device)
        image_embedding = self._clip_model.get_image_features(**inputs)
        image_embedding = image_embedding.detach().cpu()[0]  # Convert to numpy array

        if raw:
            return image, image_raw, image_embedding
        return image, image_embedding

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Returns the output image dimensions."""
        return (self._image_size, self._image_size, 3)

# Default configuration for image generation
DEFAULT_CONFIG = {
    "stable_diffusion_id": "CompVis/stable-diffusion-v1-4",
    "num_inference_steps": 100,
    "guidance_base": 8.0,
    "guidance_tokens": 4.0,
    "image_size": 512,
    "seed": 0,
    "MODELS_CACHE_DIR": os.path.expanduser("~/.cache/huggingface/hub"),
    "output_dir": "generated_images"
}

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate an image using StableDiffusionGenerator")
    parser.add_argument("--base_prompt", type=str, required=True, help="Base prompt for image generation (e.g., 'A man walking in paris')")
    parser.add_argument("--full_prompt", type=str, required=True, help="Full prompt for image generation (e.g., 'A man walking in paris #photorealistic #cute')")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help=f"Directory to save the generated image (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG['seed']})")
    parser.add_argument("--image_size", type=int, default=DEFAULT_CONFIG["image_size"], help=f"Size of the generated image (default: {DEFAULT_CONFIG['image_size']})")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_CONFIG["num_inference_steps"], help=f"Number of inference steps (default: {DEFAULT_CONFIG['num_inference_steps']})")
    parser.add_argument("--guidance_base", type=float, default=DEFAULT_CONFIG["guidance_base"], help=f"Guidance scale for base prompt (default: {DEFAULT_CONFIG['guidance_base']})")
    parser.add_argument("--guidance_tokens", type=float, default=DEFAULT_CONFIG["guidance_tokens"], help=f"Guidance scale for full prompt tokens (default: {DEFAULT_CONFIG['guidance_tokens']})")

    args = parser.parse_args()

    # Print the configuration being used
    print("Using configuration:")
    print(f"  base_prompt: '{args.base_prompt}'")
    print(f"  full_prompt: '{args.full_prompt}'")
    print(f"  stable_diffusion_id: {DEFAULT_CONFIG['stable_diffusion_id']}")
    print(f"  num_inference_steps: {args.num_inference_steps}")
    print(f"  guidance_scale: {args.guidance_base}")
    print(f"  image_size: {args.image_size}")
    print(f"  seed: {args.seed}")
    print(f"  output_dir: {args.output_dir}")
    
    # Initialize the generator with provided parameters
    generator = StableDiffusionGenerator(
        DEFAULT_CONFIG["stable_diffusion_id"],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_base,
        image_size=args.image_size,
        seed=args.seed,
        MODELS_CACHE_DIR=DEFAULT_CONFIG["MODELS_CACHE_DIR"]
    )

    # Generate the image (temporarily using only StableDiffusionGenerator with full_prompt)
    image, _ = generator.sample(args.full_prompt)
    
    # NOTE: We're temporarily using StableDiffusionGenerator instead of DoubleGuidanceStableDiffusionGenerator
    # The base_prompt and guidance_tokens parameters are accepted but not used in this version

    # Create the output directory if it doesn’t exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a filename that includes prompt and parameters
    # Sanitize the prompt for filename use
    sanitized_prompt = args.full_prompt.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized_prompt = ''.join(c for c in sanitized_prompt if c.isalnum() or c in '_-#')[:50]  # Limit length
    
    filename = f"{sanitized_prompt}_guidance{args.guidance_base}_tokens{args.guidance_tokens}_steps{args.num_inference_steps}.png"
    
    # Save the image with the descriptive filename
    image_path = os.path.join(args.output_dir, filename)
    Image.fromarray(image).save(image_path)
    print(f"Image saved to {image_path}")
