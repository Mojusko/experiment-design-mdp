import logging
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# from aesthetics_prompt_optimizer.constants import MODELS_CACHE_DIR
# from aesthetics_prompt_optimizer.environments.text2image.text2image_env_components import (
#     Text2ImageGenerator,
#)

logger = logging.getLogger(__name__)


class StableDiffusionGenerator():
    def __init__(
        self,
        stable_diffusion_id: str,
        num_inference_steps: int = 50, 
        guidance_scale: float = 7,
        image_height : int = 256, 
        image_width: int = 256,
        seed: int = 0,
        device: str = 'cpu',
   		MODELS_CACHE_DIR: str = '/tmp/models_cache_dir/'

    ) -> None:
        """An implementation of stable diffusion's text2image generator using the transformers package.

        Args:
            stable_diffusion_id (str): The stable diffusion's model identifier.
            num_inference_steps (int): The number of inverse diffusion steps to run during sampling.
            guidance_scale (float): The guidance scale for text conditioning.
            seed (int): The random seed used for sampling.
            use_cuda (bool): Whether to use GPU's or not.
        """
        clip_model_id = "openai/clip-vit-large-patch14"  # All stable diffusion models rely on VIT-14
        self.seed = seed
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self._tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR))
        self._processor = CLIPProcessor.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR))
        self._model = CLIPModel.from_pretrained(clip_model_id, cache_dir=str(MODELS_CACHE_DIR)).to(self.device)
        
        self._unet = UNet2DConditionModel.from_pretrained(
            stable_diffusion_id, subfolder="unet", cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        
        self._vae = AutoencoderKL.from_pretrained(
            stable_diffusion_id, subfolder="vae", cache_dir=str(MODELS_CACHE_DIR)
        ).to(self.device)
        
        self._scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        
        self._generator = (
            torch.manual_seed(self.seed) if self.seed else None
        )  # Seed generator to create the inital latent noise

        if (image_width, image_height) not in [(512, 512), (256, 256)]:
            logger.info(
                f"The requested image size is not guaranteed to generate good quility images. \
                Try 512x512 or 256x256 for higher quality image sampling"
            )
        self._image_size = (image_width, image_height)

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]], normalize: bool = False) -> np.ndarray:
        """Computes the embeddings of a text string using the CLIP.

        The features correspond to the projected CLIP embeddings, which is the projected [EOS]
        token feature at the highest level of the text transformer.

        Args:
            text (Union[str, List[str]]): The input text.
            normalize (bool): Whether to normalize the embedding or not.

        Returns:
            np.ndarray: The text embeddings.
        """
        if isinstance(text, str):
            text = [text]
        text_input = self._tokenizer(
            text,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        feat = self._model.get_text_features(**text_input)  # projected CLIP embeddings
        feat = feat.detach().cpu().numpy()
        if normalize:
            feat /= np.linalg.norm(feat, axis=-1)[:, np.newaxis]
        return feat.squeeze()

    @torch.no_grad()
    def encode_image(self, image: Union[np.ndarray, List[np.ndarray]], normalize: bool = False) -> np.ndarray:
        """Computes the embeddings of an image using the CLIP.

        Args:
            image (np.ndarray): The input image.
            normalize (bool): Whether to normalize the embedding or not.

        Returns:
            np.ndarray: The image embeddings.
        """
        image = self._processor(text=None, images=image, return_tensors="pt")["pixel_values"]
        feat = self._model.get_image_features(image.to(self.device))
        feat = feat.detach().cpu().numpy()
        if normalize:
            feat /= np.linalg.norm(feat, axis=-1)[:, np.newaxis]
        return feat.squeeze()


    @torch.no_grad()
    def resample_random(self) -> np.ndarray:
        height, width = self.image_size[:2]
        self.latents = torch.randn(
            (1, self._unet.in_channels, height // 8, width // 8),
            generator=self._generator,
        )

    @torch.no_grad()
    def sample(self, prompt: str, raw: bool = False) -> np.ndarray:
        """Samples an image from a text prompt.

        Args:
            prompt (str): The text prompt.

        Returns:
            np.ndarray: The sampled RGB image.
        """
        text_input = self._tokenizer(
            [prompt],
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self._model.text_model(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self._tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self._model.text_model(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = self.latents
        latents = latents.to(self.device)
        self._scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self._scheduler.init_noise_sigma

        logger.debug(f"Generating an image for the prompt {prompt}")
        for t in self._scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self._scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self._unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self._vae.decode(latents).sample
 
        image_raw = image.clone()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0, ...]
        if raw:
            return image, image_raw
        else:
            return image

    @property
    def embeddings_size(self) -> Tuple[int]:
        """Returns the CLIP model's vit-large-patch14 embedding size.

        Returns:
            Tuple[int]: The shape of the embeddings.
        """
        return (768,)

    @property
    def image_size(self) -> Tuple[int]:
        """Returns the sampled image size.

        Returns:
            Tuple[int]: The RGB image size, (length, width, 3)
        """

        return (self._image_size[0], self._image_size[1], 3)
    
if __name__ == "__main__":
    image_generator = StableDiffusionGenerator("CompVis/stable-diffusion-v1-4", device = 'cuda')
    image_generator.resample_random()

    # generate 9 images and plot them in a grid 
    images = []
    for i in range(9):
        image = image_generator.sample("A plate with")
        images.append(image)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
    plt.show()
    