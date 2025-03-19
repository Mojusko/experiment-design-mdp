import logging
from PIL import Image
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

#### Original Class: StableDiffusionGenerator
#class StableDiffusionGenerator():
#    def __init__(
#        self,
#        stable_diffusion_id: str,
#        num_inference_steps: int = 100, 
#        guidance_scale: float = 15,
#        image_size: int = 512, 
#        seed: int = 0,
#        MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/'
#    ) -> None:
#        """An implementation of stable diffusion's text2image generator.
#        
#        Args:
#            stable_diffusion_id (str): The stable diffusion's model identifier.
#            num_inference_steps (int): The number of denoising steps.
#            guidance_scale (float): The guidance scale for classifier-free guidance.
#            image_size (int): Size of generated images.
#            seed (int): Random seed for reproducibility.
#            MODELS_CACHE_DIR (str): Directory to cache the downloaded models.
#        """
#        self.seed = seed
#        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#        self.num_inference_steps = num_inference_steps
#        self.guidance_scale = guidance_scale
#
#        # Add CLIP model for image embeddings
#        self._clip_model = CLIPModel.from_pretrained(
#            "openai/clip-vit-large-patch14",
#            cache_dir=MODELS_CACHE_DIR
#        ).to(self.device)
#        self._clip_processor = CLIPProcessor.from_pretrained(
#            "openai/clip-vit-large-patch14",
#            cache_dir=MODELS_CACHE_DIR
#        )
#
#        # Load tokenizer and text encoder from the SD model
#        self._tokenizer = CLIPTokenizer.from_pretrained(
#            stable_diffusion_id,
#            subfolder="tokenizer",
#            cache_dir=str(MODELS_CACHE_DIR)
#        )
#        
#        self._text_encoder = CLIPTextModel.from_pretrained(
#            stable_diffusion_id,
#            subfolder="text_encoder",
#            cache_dir=str(MODELS_CACHE_DIR)
#        ).to(self.device)
#        
#        # Load UNet for denoising
#        self._unet = UNet2DConditionModel.from_pretrained(
#            stable_diffusion_id,
#            subfolder="unet",
#            cache_dir=str(MODELS_CACHE_DIR)
#        ).to(self.device)
#        
#        # Load VAE for image encoding/decoding
#        self._vae = AutoencoderKL.from_pretrained(
#            stable_diffusion_id,
#            subfolder="vae",
#            cache_dir=str(MODELS_CACHE_DIR)
#        ).to(self.device)
#        
#        # Setup noise scheduler
#        self._scheduler = LMSDiscreteScheduler(
#            beta_start=0.00085,
#            beta_end=0.012,
#            beta_schedule="scaled_linear",
#            num_train_timesteps=1000
#        )
#        
#        self.seed()
#
#        self._image_size = image_size
#        self.latents = None
#
#    def seed_generator(self) -> None:
#        # Setup random generator
#        if self.seed:
#            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
#        else:
#            self._generator = torch.Generator(device=self.device)
#            self._generator.seed()  # Ensure random initialization even without specific seed
#
#    @torch.no_grad()
#    def resample_random(self) -> None:
#        """Generates new random latents for image generation."""
#        latents_height = self._image_size // 8
#        latents_width = self._image_size // 8
#        self.latents = torch.randn(
#            (1, self._unet.in_channels, latents_height, latents_width),
#            generator=self._generator,
#            device=self.device
#        )
#
#    @torch.no_grad()
#    def sample(self, prompt: str, raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
#        """Generates an image from a text prompt.
#
#        Args:
#            prompt (str): The text prompt to generate an image from.
#            raw (bool): If True, returns both processed and raw image tensors.
#
#        Returns:
#            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]: Generated image(s) and text embeddings
#        """
#
#        self.seed_generator()
#
#        if self.latents is None:
#            self.resample_random()
#
#        # Encode the prompt
#        text_input = self._tokenizer(
#            [prompt],
#            padding="max_length",
#            max_length=self._tokenizer.model_max_length,
#            truncation=True,
#            return_tensors="pt"
#        )
#
#        # Get text embeddings
#        text_embeddings = self._text_encoder(text_input.input_ids.to(self.device))[0]
#
#        # Create unconditioned embeddings for classifier-free guidance
#        max_length = text_input.input_ids.shape[-1]
#        uncond_input = self._tokenizer(
#            [""], padding="max_length", max_length=max_length, return_tensors="pt"
#        )
#        uncond_embeddings = self._text_encoder(uncond_input.input_ids.to(self.device))[0]
#        
#        # Concatenate for classifier-free guidance
#        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#        
#        # Prepare latents
#        latents = self.latents.to(self.device)
#        self._scheduler.set_timesteps(self.num_inference_steps)
#        latents = latents * self._scheduler.init_noise_sigma
#
#        # Denoising loop
#        for t in self._scheduler.timesteps:
#            # Expand latents for classifier-free guidance
#            latent_model_input = torch.cat([latents] * 2)
#            latent_model_input = self._scheduler.scale_model_input(latent_model_input, timestep=t)
#
#            # Predict noise residual
#            noise_pred = self._unet(
#                latent_model_input,
#                t,
#                encoder_hidden_states=text_embeddings
#            ).sample
#
#            # Perform guidance
#            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
#
#            # Compute previous noisy sample
#            latents = self._scheduler.step(noise_pred, t, latents).prev_sample
#
#        # Decode latents to image
#        latents = 1 / 0.18215 * latents
#        image = self._vae.decode(latents).sample
#        
#        image_raw = image.clone()
#
#        # Process image for output
#        image = (image / 2 + 0.5).clamp(0, 1)
#        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#        image = (image * 255).round().astype("uint8")[0]
#
#        # Convert to PIL Image and get CLIP embedding
#        pil_image = Image.fromarray(image)
#        inputs = self._clip_processor(
#            images=pil_image, 
#            return_tensors="pt"
#        ).to(self.device)
#        image_embedding = self._clip_model.get_image_features(**inputs)
#        image_embedding = image_embedding.detach().cpu()[0]  # Convert to numpy array
#
#        if raw:
#            return image, image_raw, image_embedding
#        return image, image_embedding
#
#    @property
#    def image_size(self) -> Tuple[int, int, int]:
#        """Returns the output image dimensions.
#
#        Returns:
#            Tuple[int, int, int]: (height, width, channels)
#        """
#        return (self._image_size, self._image_size, 3)
#
### New Class: DoubleGuidanceStableDiffusionGenerator
class DoubleGuidanceStableDiffusionGenerator():
    def __init__(
        self,
        stable_diffusion_id: str,
        num_inference_steps: int = 100,
        guidance_base: float = 8,      # Guidance for base prompt
        guidance_tokens: float = 4,    # Guidance for full prompt
        image_size: int = 512,
        seed: int = 0,
        MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/'
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

# Example usage
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Using the original class
    generator = StableDiffusionGenerator(
        "CompVis/stable-diffusion-v1-4",
        seed=None,
        MODELS_CACHE_DIR=cache_dir
    )
    image, _ = generator.sample("A cat sitting on a windowsill")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Using the new class
    image_generator = DoubleGuidanceStableDiffusionGenerator(
        "CompVis/stable-diffusion-v1-4",
        guidance_base=7.5,    # Strong influence for base prompt
        guidance_tokens=2.5,  # Reduced influence for full prompt
        seed=None,
        MODELS_CACHE_DIR=cache_dir
    )
    base_prompt = "A man walking in paris"
    full_prompt = "A man walking in paris #photorealistic #cute"
    image, _ = image_generator.sample(base_prompt, full_prompt)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
