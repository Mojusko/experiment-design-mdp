import logging
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)

class StableDiffusionGenerator():
    def __init__(
        self,
        stable_diffusion_id: str,
        num_inference_steps: int = 100, 
        guidance_scale: float = 7,
        image_height: int = 512, 
        image_width: int = 512,
        seed: int = 0,
        
        MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/'
    ) -> None:
        """An implementation of stable diffusion's text2image generator.
        
        Args:
            stable_diffusion_id (str): The stable diffusion's model identifier.
            num_inference_steps (int): The number of denoising steps.
            guidance_scale (float): The guidance scale for classifier-free guidance.
            image_height (int): Height of generated images.
            image_width (int): Width of generated images.
            seed (int): Random seed for reproducibility.
            device (str): Device to run the model on ('cpu' or 'cuda').
            MODELS_CACHE_DIR (str): Directory to cache the downloaded models.
        """
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

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
        
        # Setup random generator
        if self.seed:
            self._generator = torch.manual_seed(self.seed)
        else:
            self._generator = None
            torch.seed()  # Ensure random initialization even without specific seed

        if (image_width, image_height) not in [(512, 512), (256, 256)]:
            logger.info(
                "The requested image size is not guaranteed to generate good quality images. "
                "Try 512x512 or 256x256 for higher quality image sampling"
            )
        self._image_size = (image_width, image_height)
        self.latents = None

    @torch.no_grad()
    def resample_random(self) -> None:
        """Generates new random latents for image generation."""
        latents_height = self._image_size[0] // 8
        latents_width = self._image_size[1] // 8
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
            Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]: Generated image(s)
        """
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

        if raw:
            return image, image_raw
        return image

    @property
    def image_size(self) -> Tuple[int, int, int]:
        """Returns the output image dimensions.

        Returns:
            Tuple[int, int, int]: (height, width, channels)
        """
        return (self._image_size[0], self._image_size[1], 3)

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Properly expand home directory in cache path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    image_generator = StableDiffusionGenerator(
        "CompVis/stable-diffusion-v1-4",
        
        seed=None,  # Allow for random generation
        MODELS_CACHE_DIR=cache_dir
    )

    # Generate 9 different images
    images = []
    prompt = "A beautiful plate with fresh fruits and vegetables, photorealistic"
    
    for i in range(2):
        image_generator.resample_random()  # Reset latents for each generation
        image = image_generator.sample(prompt)
        images.append(image)

    # Display the images in a 3x3 grid
    fig, axs = plt.subplots(1, 2, figsize=(4, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
