import numpy as np
import json
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig
# Removed top-level import causing circular dependency
# from experiments.llm.image_generator import StableDiffusionGenerator, DoubleGuidanceStableDiffusionGenerator, _get_seed_from_prompt
# Removed unused import causing circular dependency
# from doexpy.env.llm import create_prompt_from_tokens
import hashlib
import types

def _convert_to_serializable(obj):
    """Convert numpy arrays, torch tensors, and other non-serializable objects to Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        try:
            return _convert_to_serializable(obj.__dict__)
        except:
            return str(obj)  # Fallback to string representation
    else:
        return obj

class BaseSaver(ABC):
    """Base class for all savers with simplified interface."""

    def __init__(self, cfg: DictConfig = None, **kwargs):
        self.cfg = cfg # Store the full config if provided
        self.params = kwargs.get('params', {})
        self.scorer_model = kwargs.get('scorer_model')
        self.results_dir = kwargs.get('results_dir')
        self.experiment_id = kwargs.get('experiment_id')
        self.skip_existing = kwargs.get('skip_existing', False)

    def get_output_path(self, filename=None):
        """Get the output path with experiment_id if provided."""
        # Use instance filename if none provided
        filename = filename or self.params.get('filename')
        if not filename:
            raise ValueError("Filename must be provided either in params or as an argument")
            
        # Build filename with experiment_id if provided
        if self.experiment_id:
            # Insert experiment_id before file extension
            name, ext = os.path.splitext(filename)
            filename = f"{name}-{self.experiment_id}{ext}"
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Return full path
        path = os.path.join(self.results_dir, filename)
        
        # Check if file exists and skip_existing is True
        if self.skip_existing and os.path.exists(path):
            return None
            
        return path
        
    @abstractmethod
    def save_result(self, results):
        """Save experiment results to disk."""
        pass

class MetricsSaver(BaseSaver):
    """Saves experiment metrics to a JSON file."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_keys = self.params.get('metrics_keys', [])
        # Keys to exclude (image generation metrics should be handled by ImageGenerationSaver)
        self.excluded_keys = self.params.get('excluded_keys', [
            'best_image_score', 'worst_image_score', 'avg_top_image_score', 
            'image_generation'
        ])

    def save_result(self, results):
        file_path = self.get_output_path(self.params.get('filename', 'metrics.json'))
        
        # Get metrics from results
        metrics_dict = {}
        
        # If specific metrics keys are provided, only include those
        if self.metrics_keys:
            for key in self.metrics_keys:
                if key in results.metrics:
                    metrics_dict[key] = results.metrics[key]
                else:
                    print(f"Warning: Requested metric '{key}' not found in results")
        else:
            # Otherwise, include all metrics except excluded ones
            metrics_dict = {k: v for k, v in results.metrics.items() 
                           if k not in self.excluded_keys}
            
        # Convert non-serializable objects to JSON-compatible types
        serializable_dict = _convert_to_serializable(metrics_dict)
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_dict, f, indent=2)
            print(f"Saved metrics to {file_path}")
        except TypeError as e:
            print(f"Error saving metrics: {e}")

class ImageGenerationSaver(BaseSaver):
    """Saves generated images and related metrics from image generation tests."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8)
        self.seed = self.params.get('seed', 12)
        self.debug_mode = self.params.get('debug_mode', False)
        self.image_size = self.params.get('image_size', 512)
        self.num_inference_steps = self.params.get('num_inference_steps', 100)
        self.base_prompt = self.params.get('base_prompt', '')  # Extract base_prompt, default to empty string
        self.add_image_score = self.params.get('add_image_score', False)  # Whether to add image scores
        self.metrics_filename = self.params.get('metrics_filename', 'image_metrics.json')
        
    def save_result(self, results):
        """Save the results to a JSON file and generate images if image data is present
        
        Args:
            results: ExperimentResults object with results to save
        """
        # Create images subdirectory within results directory
        images_dir = os.path.join(self.results_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Check if there's image generation data in the results
        if "image_generation" in results.metrics:
            # Extract image generation data
            image_gen_data = results.metrics["image_generation"]
            best_prompts = image_gen_data["best_prompts"]
            best_scores = image_gen_data["best_scores"]
            worst_prompts = image_gen_data["worst_prompts"]
            worst_scores = image_gen_data["worst_scores"]
            
            # Generate images
            self._generate_images(best_prompts, best_scores, worst_prompts, worst_scores, images_dir)
            
            # Save image metrics separately
            self._save_image_metrics(results)
    
    def _save_image_metrics(self, results):
        """Save image-specific metrics to a separate JSON file"""
        # Get the image metrics
        image_metrics = {}
        
        # Include relevant image metrics
        for key in ['best_image_score', 'worst_image_score', 'avg_top_image_score']:
            if key in results.metrics:
                image_metrics[key] = results.metrics[key]
        
        # Save the image generation summary stats
        if 'image_generation' in results.metrics:
            # Include counts and score ranges but not the full prompts list
            image_gen = results.metrics['image_generation']
            image_metrics['best_prompts_count'] = len(image_gen.get('best_prompts', []))
            image_metrics['worst_prompts_count'] = len(image_gen.get('worst_prompts', []))
            
            if image_gen.get('best_scores'):
                image_metrics['best_scores_range'] = [
                    min(image_gen['best_scores']), 
                    max(image_gen['best_scores'])
                ]
                
            if image_gen.get('worst_scores'):
                image_metrics['worst_scores_range'] = [
                    min(image_gen['worst_scores']), 
                    max(image_gen['worst_scores'])
                ]
        
        # Save to file
        if image_metrics:
            file_path = self.get_output_path(self.metrics_filename)
            serializable_dict = _convert_to_serializable(image_metrics)
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(serializable_dict, f, indent=2)
                print(f"Saved image metrics to {file_path}")
            except TypeError as e:
                print(f"Error saving image metrics: {e}")
    
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

        # Import generator classes and seed function locally to avoid circular import
        from experiments.llm.image_generator import StableDiffusionGenerator, DoubleGuidanceStableDiffusionGenerator, _get_seed_from_prompt

        # Initialize image generator with debug settings if needed
        generator = StableDiffusionGenerator(
        #generator = DoubleGuidanceStableDiffusionGenerator(
            "CompVis/stable-diffusion-v1-4",
            MODELS_CACHE_DIR=os.path.expanduser("~/.cache/huggingface/hub"),
            image_size=self.image_size,
            num_inference_steps=self.num_inference_steps,
            #seed=self.seed
            seed=_get_seed_from_prompt(self.base_prompt),
        )
        
        # Generate images for the best prompts
        best_generated_images = []
        best_image_scores = []
        
        # Print debug info
        if self.debug_mode:
            print(f"DEBUG MODE: Generating smaller images ({self.image_size}x{self.image_size}) with fewer steps ({self.num_inference_steps})")
        
        print("Generating images for BEST prompts:")
        for i, (full_prompt, score) in enumerate(zip(best_prompts, best_scores)):
            print(f"Generating best image {i+1}/{len(best_prompts)} for prompt: {full_prompt}")
            # Temporarily using only full_prompt with StableDiffusionGenerator
            image, image_embedding = generator.sample(full_prompt)
            # Original: image, image_embedding = generator.sample(self.base_prompt, full_prompt)
            
            # Calculate image-based aesthetics score
            if self.add_image_score:
                # Process embedding: unsqueeze, normalize, convert to double, and move to correct device
                image_embedding = image_embedding.unsqueeze(0)
                image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)  # First L2 normalization
                image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)  # Second L2 normalization
                image_embedding = image_embedding.to(self.scorer_model.weight.device).double()
                
                # Score the embedding
                image_score = self.scorer_model.score_embedding(image_embedding).item()
                best_image_scores.append(image_score)
            else:
                image_score = None
            
            # Save the image with both scores in filename
            score_text = f"prompt_{score:.4f}"
            if image_score is not None:
                score_text += f"_image_{image_score:.4f}"
            img_path = os.path.join(images_dir, f"best_{i+1}_{score_text}.png")
            Image.fromarray(image).save(img_path)
            
            best_generated_images.append(image)
        
        # Generate images for the worst prompts
        worst_generated_images = []
        worst_image_scores = []
        
        print("\nGenerating images for WORST prompts:")
        for i, (full_prompt, score) in enumerate(zip(worst_prompts, worst_scores)):
            print(f"Generating worst image {i+1}/{len(worst_prompts)} for prompt: {full_prompt}")
            # Temporarily using only full_prompt with StableDiffusionGenerator
            image, image_embedding = generator.sample(full_prompt)
            # Original: image, image_embedding = generator.sample(self.base_prompt, full_prompt)
            
            # Calculate image-based aesthetics score
            if self.add_image_score and hasattr(self.scorer_model, 'score_embedding'):
                # Process embedding: unsqueeze, normalize, convert to double, and move to correct device
                image_embedding = image_embedding.unsqueeze(0)
                image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)  # First L2 normalization
                image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)  # Second L2 normalization
                image_embedding = image_embedding.to(self.scorer_model.weight.device).double()
                
                # Score the embedding
                image_score = self.scorer_model.score_embedding(image_embedding).item()
                worst_image_scores.append(image_score)
            else:
                image_score = None
            
            # Save the image with both scores in filename
            score_text = f"prompt_{score:.4f}"
            if image_score is not None:
                score_text += f"_image_{image_score:.4f}"
            img_path = os.path.join(images_dir, f"worst_{i+1}_{score_text}.png")
            Image.fromarray(image).save(img_path)
            
            worst_generated_images.append(image)
        
        # Create a summary image with all generated images and their scores
        # Calculate rows needed (2 rows: best and worst)
        n_cols = max(len(best_prompts), len(worst_prompts))
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        # Image scores are already calculated during image generation
        
        # Plot best images in the first row
        for i, (img, prompt_score, full_prompt) in enumerate(zip(best_generated_images, best_scores, best_prompts)):
            axes[0, i].imshow(img)
            title = f"Best {i+1}: Prompt {prompt_score:.4f}"
            if i < len(best_image_scores):
                title += f"\nImage {best_image_scores[i]:.4f}"
            axes[0, i].set_title(title)
            axes[0, i].set_xlabel(full_prompt, fontsize=8)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
        
        # Hide any unused subplots in first row
        for i in range(len(best_generated_images), n_cols):
            axes[0, i].axis('off')
        
        # Plot worst images in the second row
        for i, (img, prompt_score, full_prompt) in enumerate(zip(worst_generated_images, worst_scores, worst_prompts)):
            axes[1, i].imshow(img)
            title = f"Worst {i+1}: Prompt {prompt_score:.4f}"
            if i < len(worst_image_scores):
                title += f"\nImage {worst_image_scores[i]:.4f}"
            axes[1, i].set_title(title)
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
            f.write("Rank\tPrompt Score\tImage Score\tPrompt\n")
            for i, (prompt_score, prompt) in enumerate(zip(best_scores, best_prompts)):
                image_score = best_image_scores[i] if i < len(best_image_scores) else "N/A"
                f.write(f"{i+1}\t{prompt_score:.6f}\t{image_score}\t{prompt}\n")
            
            f.write("\nWORST PROMPTS:\n")
            f.write("Rank\tPrompt Score\tImage Score\tPrompt\n")
            for i, (prompt_score, prompt) in enumerate(zip(worst_scores, worst_prompts)):
                image_score = worst_image_scores[i] if i < len(worst_image_scores) else "N/A"
                f.write(f"{i+1}\t{prompt_score:.6f}\t{image_score}\t{prompt}\n")


class LearnedEstimatorSaver(BaseSaver):
    """Saves the learned estimator theta vector to a file."""
    
    def save_result(self, results):
        file_path = self.get_output_path(self.params.get('filename', 'estimator.pt'))
        
        # Get theta from results
        theta = results.get_theta()
        
        # If theta is available, save it
        if theta is not None:
            try:
                torch.save(theta, file_path)
                print(f"Saved estimator theta (shape: {theta.shape}) to {file_path}")
            except Exception as e:
                print(f"Error saving estimator: {e}")
        else:
            print("Error: No theta vector available in estimator")

class VisitsSaver(BaseSaver):
    """Saves exploration visits to a file."""
    
    def save_result(self, results):
        file_path = self.get_output_path(self.params.get('filename', 'visits.pkl'))
        
        # Get visits from results
        visits = results.visits
        
        if visits is None:
            print("Error: No visits available in results")
            return
        
        # Save visits to file
        try:
            torch.save(visits, file_path)
            print(f"Saved visits to {file_path}")
        except Exception as e:
            print(f"Warning when saving visits: {e}")
            try:
                # Try converting to a serializable format
                converted_visits = _convert_to_serializable(visits)
                torch.save(converted_visits, file_path)
                print(f"Saved converted visits to {file_path}")
            except Exception as e2:
                print(f"Failed to save visits: {e2}")

class ConfSaver(BaseSaver):
    """Saves the experiment configuration to a YAML file."""

    def save_result(self, results):
        """Saves the configuration to a YAML file."""
        file_path = self.get_output_path(self.params.get('filename', 'config_resolved.yaml'))
        if file_path is None: # Skip if file exists and skip_existing is True
            print(f"Skipping saving config as it already exists.")
            return

        # Get the raw dictionary config from the experiment result metadata
        config_dict = results.metadata.get('config_dict')
        
        if config_dict is None:
            print("Error: No configuration data in results metadata. Add it with results.add_metadata('config_dict', config_dict)")
            return

        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print(f"Saved configuration to {file_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
