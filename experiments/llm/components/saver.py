import numpy as np
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
from abc import ABC, abstractmethod
from experiments.llm.image_generator import StableDiffusionGenerator, DoubleGuidanceStableDiffusionGenerator
from doexpy.env.llm import create_prompt_from_tokens

class BaseSaver(ABC):
    def __init__(self, scorer_model=None, params=None, results_dir=None, experiment_id=None):
        self.params = params or {}
        self.scorer_model = scorer_model
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        
    @abstractmethod
    def save_result(self, result_dict):
        pass

class FileSaver(BaseSaver):
    def __init__(self, scorer_model=None, params=None, results_dir=None, experiment_id=None):
        super().__init__(scorer_model, params, results_dir, experiment_id)
        self.filename = self.params.get('filename', 'metrics.json')

    def save_result(self, result_dict):
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Build filename with experiment_id if provided
        filename = self.filename
        if self.experiment_id:
            # Insert experiment_id before file extension
            name, ext = os.path.splitext(self.filename)
            filename = f"{name}-{self.experiment_id}{ext}"
        
        # Build full path
        file_path = os.path.join(self.results_dir, filename)
        
        # Dump the entire result_dict to the specified file in JSON format
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved result to {file_path}:")
        print(json.dumps(result_dict, indent=2))

class ImageGenerationSaver(BaseSaver):
    def __init__(self, scorer_model=None, params=None, results_dir=None, experiment_id=None):
        super().__init__(scorer_model, params, results_dir, experiment_id)
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8)
        self.debug_mode = self.params.get('debug_mode', False)
        self.image_size = self.params.get('image_size', 512)
        self.num_inference_steps = self.params.get('num_inference_steps', 100)
        self.base_prompt = self.params.get('base_prompt', '')  # Extract base_prompt, default to empty string
        
    def save_result(self, result_dict):
        """Save the results to a JSON file and generate images if image data is present
        
        Args:
            result_dict: Dictionary with results to save
        """
        # Create images subdirectory within results directory
        images_dir = os.path.join(self.results_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Check if there's image generation data in the results
        if "image_generation" in result_dict:
            best_prompts = result_dict["image_generation"]["best_prompts"]
            best_scores = result_dict["image_generation"]["best_scores"]
            worst_prompts = result_dict["image_generation"]["worst_prompts"]
            worst_scores = result_dict["image_generation"]["worst_scores"]
            self._generate_images(best_prompts, best_scores, worst_prompts, worst_scores, images_dir)
    
    def _generate_images(self, best_prompts, best_scores, worst_prompts, worst_scores, images_dir):
        """Generate images from lists of best and worst prompts (internal method)
        
        Args:
            best_prompts: List of best prompts to generate images for
            best_scores: List of scores for each best prompt
            worst_prompts: List of worst prompts to generate images for
            worst_scores: List of scores for each worst prompt
            images_dir: Directory to save images to
        """
        # Create experiment-specific subdirectory if experiment_id is provided
        if self.experiment_id:
            images_dir = os.path.join(images_dir, self.experiment_id)
            os.makedirs(images_dir, exist_ok=True)
        
        # Initialize image generator with debug settings if needed
        #generator = StableDiffusionGenerator(
        generator = DoubleGuidanceStableDiffusionGenerator(
            "CompVis/stable-diffusion-v1-4",
            MODELS_CACHE_DIR=os.path.expanduser("~/.cache/huggingface/hub"),
            image_size=self.image_size,
            num_inference_steps=self.num_inference_steps
        )
        
        # Generate images for the best prompts
        best_generated_images = []
        
        # Print debug info
        if self.debug_mode:
            print(f"DEBUG MODE: Generating smaller images ({self.image_size}x{self.image_size}) with fewer steps ({self.num_inference_steps})")
        
        print("Generating images for BEST prompts:")
        for i, (full_prompt, score) in enumerate(zip(best_prompts, best_scores)):
            print(f"Generating best image {i+1}/{len(best_prompts)} for full_prompt: {full_prompt}")
            image, _ = generator.sample(self.base_prompt, full_prompt, raw=False)
            
            # Save the image
            img_path = os.path.join(images_dir, f"best_{i+1}_score_{score:.4f}.png")
            Image.fromarray(image).save(img_path)
            
            best_generated_images.append(image)
        
        # Generate images for the worst prompts
        worst_generated_images = []
        
        print("\nGenerating images for WORST prompts:")
        for i, (prompt, score) in enumerate(zip(worst_prompts, worst_scores)):
            print(f"Generating worst image {i+1}/{len(worst_prompts)} for prompt: {prompt}")
            image, _ = generator.sample(prompt, raw=False)
            
            # Save the image
            img_path = os.path.join(images_dir, f"worst_{i+1}_score_{score:.4f}.png")
            Image.fromarray(image).save(img_path)
            
            worst_generated_images.append(image)
        
        # Create a summary image with all generated images and their scores
        # Calculate rows needed (2 rows: best and worst)
        n_cols = max(len(best_prompts), len(worst_prompts))
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        # Plot best images in the first row
        for i, (img, score, full_prompt) in enumerate(zip(best_generated_images, best_scores, best_prompts)):
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Best {i+1}: {score:.4f}")
            axes[0, i].set_xlabel(prompt, fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
        
        # Hide any unused subplots in first row
        for i in range(len(best_generated_images), n_cols):
            axes[0, i].axis('off')
        
        # Plot worst images in the second row
        for i, (img, score, full_prompt) in enumerate(zip(worst_generated_images, worst_scores, worst_prompts)):
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Worst {i+1}: {score:.4f}")
            axes[1, i].set_xlabel(full_prompt, fontsize=8)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # Hide any unused subplots in second row
        for i in range(len(worst_generated_images), n_cols):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(images_dir, "summary.png")
        plt.savefig(summary_path)
        plt.close()
        
        # Save scores and prompts to a text file
        with open(os.path.join(images_dir, "results.txt"), "w") as f:
            f.write("BEST PROMPTS:\n")
            f.write("Rank\tScore\tPrompt\n")
            for i, (score, prompt) in enumerate(zip(best_scores, best_prompts)):
                f.write(f"{i+1}\t{score:.6f}\t{prompt}\n")
            
            f.write("\nWORST PROMPTS:\n")
            f.write("Rank\tScore\tPrompt\n")
            for i, (score, prompt) in enumerate(zip(worst_scores, worst_prompts)):
                f.write(f"{i+1}\t{score:.6f}\t{prompt}\n")
    
