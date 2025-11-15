# Standard library imports
import json
import os
import re
import textwrap
# import types # Removed unused import
# import hashlib # Removed unused import

# Third-party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import PIL.Image
import yaml # Added yaml import for ConfSaver
from abc import ABC, abstractmethod
from omegaconf import DictConfig # Keep DictConfig for type hints
# from hydra.utils import to_absolute_path # Removed unused import

# Local imports
from doexpy.env.llm import create_prompt
# Moved imports to top level:
from experiments.llm.image_generator import StableDiffusionGenerator, _get_seed_from_prompt, DEFAULT_CONFIG
from experiments.llm.components.tester import create_dot_product_model_from_estimator # Add this import


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
    """
    Base class for all savers with a standardized interface.
    
    All savers handle storing experiment results in various formats
    and configurations. This provides a common interface for different
    saving strategies.
    """

    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None,
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False,
                 **kwargs): # Add kwargs to accept unused arguments
        """
        Initialize the base saver with common parameters.
        Args:
            env: The environment object
            embedder: The text/image embedder
            params: Saver-specific parameters from Hydra config
            scorer_model: Model used for scoring outputs
            results_dir: Directory to save results
            experiment_id: Optional ID to identify this experiment run
            skip_existing: If True, skip saving if output file exists
        """
        self.env = env
        self.embedder = embedder
        self.params = params if params is not None else {}
        self.scorer_model = scorer_model
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.skip_existing = skip_existing
        # self.cfg = cfg # REMOVED cfg storage

    def get_output_path(self, filename=None):
        """
        Get the output path with experiment_id if provided.
        
        Args:
            filename: Optional filename to use instead of the one in params
            
        Returns:
            Full path to the output file, or None if file exists and skip_existing is True
        """
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
            'best_prompt_score', 'worst_prompt_score', 'avg_top_prompt_score',
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
                    # Check context before warning to avoid benign warnings in multi-model main save
                    is_multi_model_main_save_context = False
                    # The main 'results' object will have 'scorer_model_names' in its metadata.
                    # Temporary 'results' objects for per-model saves typically won't.
                    if 'scorer_model_names' in results.metadata and \
                       isinstance(results.metadata['scorer_model_names'], list) and \
                       len(results.metadata['scorer_model_names']) > 1:
                        is_multi_model_main_save_context = True
                    
                    known_per_model_keys = ["preference_error", "cosine_error"] # Keys handled per-model

                    if not (is_multi_model_main_save_context and key in known_per_model_keys):
                        print(f"Warning: Requested metric '{key}' not found in results")
        else:
            # Otherwise, include all metrics except excluded ones
            metrics_dict = {k: v for k, v in results.metrics.items() 
                           if k not in self.excluded_keys}
            
        # Convert non-serializable objects to JSON-compatible types
        serializable_dict = _convert_to_serializable(metrics_dict)

        # Only save the file if there are metrics to save
        if not serializable_dict:
            print(f"Skipping save for {file_path} as there are no metrics to save after filtering.")
            return
        
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
        # DEFAULT_CONFIG is now imported at the top level

        self.take_best_worst_N = self.params.get('take_best_worst_N', 4)
        self.debug_mode = self.params.get('debug_mode', False)
        # Get image_size and num_inference_steps from params or DEFAULT_CONFIG
        self.image_size = self.params.get('image_size', DEFAULT_CONFIG['image_size'])
        self.num_inference_steps = self.params.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
        # Saver's own base_prompt, defaults to None to easily check if it was set.
        self.base_prompt = self.params.get('base_prompt', None)
        self.add_scores = self.params.get('add_scores', ['image', 'prompt']) # List: 'image', 'prompt'
        self.metrics_filename = self.params.get('metrics_filename', 'image_metrics.json')
        self.save_worst = False # Forcing save_worst to False as per new requirement

        # Process score_images_with_model
        raw_score_images_with_model = self.params.get('score_images_with_model', 'gt')
        if isinstance(raw_score_images_with_model, list):
            if len(raw_score_images_with_model) == 1:
                self.score_images_with_model = raw_score_images_with_model[0]
            else:
                raise ValueError(f"ImageGenerationSaver: score_images_with_model must be a string or a single-element list, got {raw_score_images_with_model}")
        elif isinstance(raw_score_images_with_model, str):
            self.score_images_with_model = raw_score_images_with_model
        else:
            raise ValueError(f"ImageGenerationSaver: score_images_with_model must be a string or a list, got {type(raw_score_images_with_model)}")

        # Process secondary_gt_prompt_score_for_model
        raw_secondary_gt_model = self.params.get('secondary_gt_prompt_score_for_model', None)
        if raw_secondary_gt_model is None:
            self.secondary_gt_prompt_score_for_model = None
        elif isinstance(raw_secondary_gt_model, list):
            if len(raw_secondary_gt_model) == 1:
                self.secondary_gt_prompt_score_for_model = raw_secondary_gt_model[0]
            else:
                raise ValueError(f"ImageGenerationSaver: secondary_gt_prompt_score_for_model must be None, a string, or a single-element list, got {raw_secondary_gt_model}")
        elif isinstance(raw_secondary_gt_model, str):
            self.secondary_gt_prompt_score_for_model = raw_secondary_gt_model
        else:
            raise ValueError(f"ImageGenerationSaver: secondary_gt_prompt_score_for_model must be None, a string, or a list, got {type(raw_secondary_gt_model)}")
        
        # prompt_ranking_model is not directly used by saver, it relies on tester's output (best_scores/worst_scores)
        
    def save_result(self, results):
        """Save the results to a JSON file and generate images if image data is present
        
        Args:
            results: ExperimentResults object with results to save
        """
        # Determine current model name for directory/file naming
        current_model_name = results.metadata.get('current_model_name', self.experiment_id or 'unknown_model')

        # Create a base directory for this model's images, e.g., results/images_sunny/
        base_images_dir = os.path.join(self.results_dir, f"images_{current_model_name}")

        # Output directory is determined by model name; feedback metadata is kept for reference only.
        output_images_dir = base_images_dir

        os.makedirs(output_images_dir, exist_ok=True)
        
        # Check if there's image generation data in the results
        if "image_generation" in results.metrics:
            # Extract image generation data
            image_gen_data = results.metrics["image_generation"]
            best_prompts = image_gen_data["best_prompts"]
            best_scores = image_gen_data["best_scores"] # These are prompt scores from the tester
            # Worst prompts are no longer processed by the tester or saver
            worst_prompts = [] 
            worst_scores = []
            
            # Generate images and potentially calculate new image scores, saving into output_images_dir
            self._generate_images(best_prompts, best_scores, worst_prompts, worst_scores, output_images_dir, results) # Pass results
            
            # Save image metrics separately, using model_specific name and potentially prompt-specific path
            self._save_image_metrics(results, current_model_name, output_images_dir)
    
    def _save_image_metrics(self, results, current_model_name_for_file, output_dir):
        """Save image-specific metrics to a separate JSON file, aligned with configuration."""
        image_metrics = {}
        image_gen_data = results.metrics.get("image_generation", {})

        # Counts are always relevant if image generation happened
        image_metrics['best_prompts_count'] = len(image_gen_data.get('best_prompts', []))
        # image_metrics['worst_prompts_count'] = len(image_gen_data.get('worst_prompts', [])) # Worst prompts not processed

        # Base prompt image scores (if generated and scored)
        if 'base_image_score' in image_gen_data and image_gen_data['base_image_score'] is not None:
            image_metrics['base_image_score'] = image_gen_data['base_image_score']
        if 'base_prompt_secondary_gt_score' in image_gen_data and image_gen_data['base_prompt_secondary_gt_score'] is not None:
            image_metrics['base_prompt_secondary_gt_score'] = image_gen_data['base_prompt_secondary_gt_score']


        # Primary prompt scores (from tester's ranking model)
        if 'prompt' in self.add_scores:
            # These keys are now best_prompt_score, worst_prompt_score, avg_top_prompt_score
            if 'best_prompt_score' in results.metrics:
                image_metrics['best_prompt_score'] = results.metrics['best_prompt_score']
            # if 'worst_prompt_score' in results.metrics: # Worst prompts not processed
            #     image_metrics['worst_prompt_score'] = results.metrics['worst_prompt_score']
            if 'avg_top_prompt_score' in results.metrics:
                image_metrics['avg_top_prompt_score'] = results.metrics['avg_top_prompt_score']
        
        # Secondary GT prompt scores (for prompts selected by tester)
        if 'prompt' in self.add_scores and self.secondary_gt_prompt_score_for_model:
            sec_ps_best = image_gen_data.get('secondary_gt_prompt_scores_best', [])
            # sec_ps_worst = image_gen_data.get('secondary_gt_prompt_scores_worst', []) # Worst prompts not processed
            # Filter out None values before calculating stats
            valid_sec_ps_best = [s for s in sec_ps_best if s is not None]
            # valid_sec_ps_worst = [s for s in sec_ps_worst if s is not None] # Worst prompts not processed

            if valid_sec_ps_best:
                image_metrics['best_secondary_gt_prompt_score'] = max(valid_sec_ps_best)
                image_metrics['avg_top_secondary_gt_prompt_score'] = sum(valid_sec_ps_best) / len(valid_sec_ps_best)
            # if valid_sec_ps_worst: # Worst prompts not processed
            #     image_metrics['worst_secondary_gt_prompt_score'] = max(valid_sec_ps_worst)


        # Actual generated image scores (if calculated)
        if 'image' in self.add_scores:
            img_scores_best = image_gen_data.get('generated_image_scores_best', [])
            # img_scores_worst = image_gen_data.get('generated_image_scores_worst', []) # Worst prompts not processed
            # Filter out None values
            valid_img_scores_best = [s for s in img_scores_best if s is not None]
            # valid_img_scores_worst = [s for s in img_scores_worst if s is not None] # Worst prompts not processed

            if valid_img_scores_best:
                image_metrics['best_generated_image_score'] = max(valid_img_scores_best)
                image_metrics['avg_top_generated_image_score'] = sum(valid_img_scores_best) / len(valid_img_scores_best)
            # if valid_img_scores_worst: # Worst prompts not processed
            #     image_metrics['worst_generated_image_score'] = max(valid_img_scores_worst)
        
        # Save to file only if there's something to save
        if image_metrics:
            # Construct model-specific metrics filename, e.g., image_metrics_sunny.json
            base_metrics_filename = self.params.get('metrics_filename', 'image_metrics.json')
            name, ext = os.path.splitext(base_metrics_filename)
            model_specific_metrics_filename = f"{name}_{current_model_name_for_file}{ext}"
            
            # Save the metrics file inside the specific output directory for this run
            file_path = os.path.join(output_dir, model_specific_metrics_filename)

            serializable_dict = _convert_to_serializable(image_metrics)
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(serializable_dict, f, indent=2)
                print(f"Saved image metrics to {file_path}")
            except TypeError as e:
                print(f"Error saving image metrics: {e}")
    
    def _generate_images(self, best_prompts, best_scores, worst_prompts, worst_scores, output_images_dir, results):
        """Generate images from lists of best and worst prompts (internal method)
        
        Args:
            best_prompts: List of best prompts to generate images for
            best_scores: List of scores for each best prompt (prompt scores from tester)
            worst_prompts: List of worst prompts to generate images for
            worst_scores: List of scores for each worst prompt (prompt scores from tester)
            output_images_dir: Directory to save images to (e.g., results/images_sunny/a_cat_sleeping)
            results: ExperimentResults object, used to access estimator for image scoring
        """
        # Determine the effective base prompt with a clear priority:
        # 1. base_prompt from the saver's own config
        # 2. base_prompt from the global environment
        if self.base_prompt is not None:
            effective_base_prompt = self.base_prompt
        else:
            effective_base_prompt = self.env.base_prompt

        # Initialize image generator using self attributes (derived from params/DEFAULT_CONFIG)
        # and specific seed logic for this saver.
        generator = StableDiffusionGenerator(
            stable_diffusion_id=DEFAULT_CONFIG['stable_diffusion_id'], # Use default model ID
            MODELS_CACHE_DIR=DEFAULT_CONFIG['MODELS_CACHE_DIR'], # Use default cache dir
            image_size=self.image_size, # Use size from __init__ (params or default)
            num_inference_steps=self.num_inference_steps, # Use steps from __init__ (params or default)
            guidance_scale=DEFAULT_CONFIG['guidance_scale'], # Use default guidance
            seed=_get_seed_from_prompt(effective_base_prompt), # Use effective base_prompt for seeding
        )
        
        # Generate images for the best prompts
        best_generated_images = []
        # best_image_scores = [] # Removed unused variable
        base_image_data = None # To store (image, prompt_score, image_score) for the base prompt

        # Print debug info
        if self.debug_mode:
            print(f"DEBUG MODE: Generating smaller images ({self.image_size}x{self.image_size}) with fewer steps ({self.num_inference_steps})")
        
        # Determine scoring model for images if 'image' in self.add_scores
        scoring_model_for_images = None
        if 'image' in self.add_scores: # This block is for SCORING GENERATED IMAGES
            if self.score_images_with_model == 'gt':
                # Assuming ImageGenerationTester runs for the first model, so GT image scoring also uses the first GT model.
                # LLMExperiment should add 'all_gt_scorer_models' to results.metadata
                available_gt_models = results.metadata.get('all_gt_scorer_models')
                if not available_gt_models or not available_gt_models[0]:
                    print("Warning: ImageGenerationSaver: Cannot score images with 'gt'. Ground truth scorer model (first model) not available in results metadata.")
                else:
                    scoring_model_for_images = available_gt_models[0]
                    first_model_name = results.metadata.get('scorer_model_names', ['N/A'])[0]
                    print(f"ImageGenerationSaver: Will calculate image scores using the ground truth scorer model ('{first_model_name}').")
            else: # It's a specific model name
                model_name_for_image_scoring = self.score_images_with_model
                experiment_model_names = results.metadata.get('scorer_model_names', [])
                
                if not experiment_model_names:
                    print(f"Warning: ImageGenerationSaver: Cannot score images with '{model_name_for_image_scoring}'. No scorer_model_names in results metadata.")
                elif model_name_for_image_scoring not in experiment_model_names:
                    print(f"Warning: ImageGenerationSaver: Image scoring model '{model_name_for_image_scoring}' not found in experiment's scorer_model list: {experiment_model_names}")
                else:
                    model_idx = experiment_model_names.index(model_name_for_image_scoring)
                    available_estimators = results.estimators # This is a list of estimator objects
                    if not available_estimators or model_idx >= len(available_estimators) or available_estimators[model_idx] is None:
                        print(f"Warning: ImageGenerationSaver: Estimator for image scoring model '{model_name_for_image_scoring}' (index {model_idx}) is not available.")
                    elif self.embedder is None: # self.embedder is from BaseSaver.__init__
                        print(f"Warning: ImageGenerationSaver: Cannot create estimator model for image scoring ('{model_name_for_image_scoring}') without an embedder instance.")
                    else:
                        try:
                            scoring_model_for_images = create_dot_product_model_from_estimator(available_estimators[model_idx], self.embedder)
                            print(f"ImageGenerationSaver: Will calculate image scores using the estimator for '{model_name_for_image_scoring}'.")
                        except Exception as e:
                            print(f"Warning: ImageGenerationSaver: Could not create model from estimator for image scoring ('{model_name_for_image_scoring}'): {e}")

        # Determine secondary GT model for prompt scoring if configured
        secondary_gt_model_for_prompts = None
        if self.secondary_gt_prompt_score_for_model and 'prompt' in self.add_scores:
            model_name_for_secondary_prompt_scoring = self.secondary_gt_prompt_score_for_model
            available_gt_models = results.metadata.get('all_gt_scorer_models')
            experiment_model_names = results.metadata.get('scorer_model_names', [])

            if not available_gt_models or not experiment_model_names:
                print(f"Warning: ImageGenerationSaver: Cannot get secondary GT prompt scores for '{model_name_for_secondary_prompt_scoring}'. GT models or names not in results metadata.")
            elif model_name_for_secondary_prompt_scoring not in experiment_model_names:
                print(f"Warning: ImageGenerationSaver: Secondary GT prompt scoring model '{model_name_for_secondary_prompt_scoring}' not found in experiment's scorer_model list: {experiment_model_names}")
            else:
                model_idx = experiment_model_names.index(model_name_for_secondary_prompt_scoring)
                if model_idx >= len(available_gt_models) or available_gt_models[model_idx] is None:
                    print(f"Warning: ImageGenerationSaver: Secondary GT scorer model '{model_name_for_secondary_prompt_scoring}' (index {model_idx}) is not available from all_gt_scorer_models list.")
                else:
                    secondary_gt_model_for_prompts = available_gt_models[model_idx]
                    print(f"ImageGenerationSaver: Will calculate secondary prompt scores using the ground truth scorer model for '{model_name_for_secondary_prompt_scoring}'.")

        # Generate base image if effective_base_prompt is set
        base_image_tuple = None # (image_array, secondary_gt_prompt_score, image_score)
        if effective_base_prompt and effective_base_prompt.strip():
            print(f"Generating base image for prompt: {effective_base_prompt}")
            base_img_array, base_img_embedding = generator.sample(effective_base_prompt, embedder=self.embedder)
            
            base_img_sec_prompt_score = None
            if secondary_gt_model_for_prompts:
                try:
                    score_tensor, _ = secondary_gt_model_for_prompts.score_prompt(effective_base_prompt)
                    base_img_sec_prompt_score = score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to get secondary GT prompt score for base prompt '{effective_base_prompt[:30]}...': {e}")

            base_img_actual_score = None
            if scoring_model_for_images:
                try:
                    img_score_tensor = scoring_model_for_images.score_embedding(base_img_embedding.to(self.embedder.device))
                    base_img_actual_score = img_score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to score base image for prompt '{effective_base_prompt[:30]}...': {e}")
            
            base_image_tuple = (base_img_array, base_img_sec_prompt_score, base_img_actual_score)
            
            # Save standalone base image
            base_filename_parts = ["base_image"]
            if 'prompt' in self.add_scores and base_img_sec_prompt_score is not None: # Using secondary GT score for filename
                base_filename_parts.append(f"sec_pscore_{base_img_sec_prompt_score:.4f}")
            if 'image' in self.add_scores and base_img_actual_score is not None:
                base_filename_parts.append(f"iscore_{base_img_actual_score:.4f}")
            base_img_filename = "_".join(base_filename_parts) + ".png"
            base_img_path = os.path.join(output_images_dir, base_img_filename) # Use output_images_dir
            PIL.Image.fromarray(base_img_array).save(base_img_path)
            print(f"Saved base image to {base_img_path}")
            
            # Store base image scores in results for _save_image_metrics
            if 'image_generation' not in results.metrics: results.metrics['image_generation'] = {}
            results.metrics['image_generation']['base_image_score'] = base_img_actual_score
            results.metrics['image_generation']['base_prompt_secondary_gt_score'] = base_img_sec_prompt_score


        actual_best_image_scores = [] # Stores actual scores of generated images
        actual_best_secondary_prompt_scores = [] # Stores secondary GT scores for best prompts
        print("Generating images for BEST prompts:")
        # best_scores contains primary prompt scores from the tester
        for i, (full_prompt, primary_prompt_score) in enumerate(zip(best_prompts, best_scores)):
            print(f"Generating best image {i+1}/{len(best_prompts)} for prompt: {full_prompt}")
            image, image_embedding = generator.sample(full_prompt, embedder=self.embedder)
            
            current_image_score = None # Actual score of this generated image
            if scoring_model_for_images: # Only if 'image' in add_scores and model is available
                try:
                    img_score_tensor = scoring_model_for_images.score_embedding(image_embedding.to(self.embedder.device))
                    current_image_score = img_score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to score image for prompt '{full_prompt[:30]}...': {e}")
            actual_best_image_scores.append(current_image_score)

            current_secondary_prompt_score = None
            if secondary_gt_model_for_prompts:
                try:
                    # score_prompt returns (score_tensor, embedding_tensor)
                    score_tensor, _ = secondary_gt_model_for_prompts.score_prompt(full_prompt)
                    current_secondary_prompt_score = score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to get secondary GT prompt score for '{full_prompt[:30]}...': {e}")
            actual_best_secondary_prompt_scores.append(current_secondary_prompt_score)

            filename_parts = [f"best_{i+1}"]
            if 'prompt' in self.add_scores:
                filename_parts.append(f"pscore_{primary_prompt_score:.4f}")
                if current_secondary_prompt_score is not None:
                    filename_parts.append(f"sec_pscore_{current_secondary_prompt_score:.4f}")
            if 'image' in self.add_scores and current_image_score is not None:
                filename_parts.append(f"iscore_{current_image_score:.4f}")
            img_filename = "_".join(filename_parts) + ".png"
            img_path = os.path.join(output_images_dir, img_filename) # Use output_images_dir
            PIL.Image.fromarray(image).save(img_path)
            best_generated_images.append(image)
        
        # Worst images are no longer processed
        worst_generated_images = []
        actual_worst_image_scores = [] 
        actual_worst_secondary_prompt_scores = []

        # Store detailed score lists in results.metrics['image_generation']
        if 'image_generation' not in results.metrics: 
            results.metrics['image_generation'] = {}
        
        results.metrics['image_generation']['generated_image_scores_best'] = actual_best_image_scores
        # results.metrics['image_generation']['generated_image_scores_worst'] = actual_worst_image_scores # Not processed
        results.metrics['image_generation']['secondary_gt_prompt_scores_best'] = actual_best_secondary_prompt_scores
        # results.metrics['image_generation']['secondary_gt_prompt_scores_worst'] = actual_worst_secondary_prompt_scores # Not processed

        # Create a summary image
        n_rows = 1 # Only one row for base (optional) + best
        num_best_to_plot = len(best_generated_images)
        n_cols = num_best_to_plot
        if base_image_tuple:
            n_cols += 1 # Add column for base image

        if n_cols == 0:
             print("No images (base or best) generated, skipping summary figure.")
             plt.close() 
             return 

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4.5*n_rows), squeeze=False) # Adjusted height slightly

        current_col_idx = 0
        if base_image_tuple:
            base_img_array, base_prompt_score, base_img_score = base_image_tuple
            ax = axes[0, current_col_idx]
            ax.imshow(base_img_array)
            ax.set_title("Base Prompt") # Title for base image without scores
            wrapped_prompt = textwrap.fill(effective_base_prompt, width=40)
            ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add a vertical separator line if there are best images to follow
            if num_best_to_plot > 0:
                ax.axvline(x=ax.get_xlim()[1], color='gray', linestyle='--', linewidth=1.5, ymin=0.05, ymax=0.95)
            
            current_col_idx +=1

        # Add a title for the "Top Generated Prompts" section if there are best images
        if num_best_to_plot > 0:
            # Determine the axes for the best images section
            first_best_ax = axes[0, current_col_idx]
            last_best_ax = axes[0, current_col_idx + num_best_to_plot - 1]
            
            # Get positions for centering the text
            x_start_fig_coord = first_best_ax.get_position().x0
            x_end_fig_coord = last_best_ax.get_position().x1
            y_pos_fig_coord = first_best_ax.get_position().y1 + 0.02 # Adjust 0.02 for spacing
            
            fig.text((x_start_fig_coord + x_end_fig_coord) / 2, y_pos_fig_coord,
                     "Top Generated Prompts",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        for i, (img, p_score, sec_p_score, full_prompt, i_score) in enumerate(zip(best_generated_images, best_scores, actual_best_secondary_prompt_scores, best_prompts, actual_best_image_scores)):
           ax = axes[0, current_col_idx + i] # current_col_idx is 1 if base_image exists, 0 otherwise
           ax.imshow(img)
           title_parts = [f"Best {i+1}"]
           if 'prompt' in self.add_scores:
               title_parts.append(f"RankScr: {p_score:.2f}")
               if sec_p_score is not None:
                   title_parts.append(f"GTScr: {sec_p_score:.2f}")
           if 'image' in self.add_scores and i_score is not None:
               title_parts.append(f"IScr: {i_score:.2f}")
           ax.set_title(" ".join(title_parts))
           wrapped_prompt = textwrap.fill(full_prompt, width=40)
           ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10)
           ax.set_xticks([])
           ax.set_yticks([])

        # Turn off any remaining axes if n_cols was larger than images plotted (e.g. base only, no best)
        for i in range(current_col_idx + num_best_to_plot, n_cols):
            axes[0, i].axis('off')


        plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.5)
        summary_path = os.path.join(output_images_dir, "summary.png") # Use output_images_dir
        plt.savefig(summary_path)
        plt.close()

        with open(os.path.join(output_images_dir, "results.txt"), "w") as f: # Use output_images_dir
            if base_image_tuple:
                _, base_prompt_score, base_img_score = base_image_tuple
                f.write("BASE PROMPT:\n")
                header_parts_base = []
                if 'prompt' in self.add_scores and self.secondary_gt_prompt_score_for_model:
                    header_parts_base.append("GTScore")
                if 'image' in self.add_scores: header_parts_base.append("ImageScore")
                header_parts_base.append("Prompt")
                f.write("\t".join(header_parts_base) + "\n")
                
                line_parts_base = []
                if 'prompt' in self.add_scores and self.secondary_gt_prompt_score_for_model:
                    line_parts_base.append(f"{base_prompt_score:.6f}" if base_prompt_score is not None else "N/A")
                if 'image' in self.add_scores:
                    line_parts_base.append(f"{base_img_score:.6f}" if base_img_score is not None else "N/A")
                line_parts_base.append(effective_base_prompt)
                f.write("\t".join(line_parts_base) + "\n\n")


            f.write("BEST PROMPTS:\n")
            header_parts = ["Rank"]
            if 'prompt' in self.add_scores:
                header_parts.append("RankScore")
                if self.secondary_gt_prompt_score_for_model: # Add header if secondary scoring is active
                    header_parts.append("GTScore")
            if 'image' in self.add_scores: header_parts.append("ImageScore")
            header_parts.append("Prompt")
            f.write("\t".join(header_parts) + "\n")

            for i, (p_score, sec_p_score, prompt, i_score) in enumerate(zip(best_scores, actual_best_secondary_prompt_scores, best_prompts, actual_best_image_scores)):
               line_parts = [str(i+1)]
               if 'prompt' in self.add_scores:
                   line_parts.append(f"{p_score:.6f}")
                   if self.secondary_gt_prompt_score_for_model:
                       line_parts.append(f"{sec_p_score:.6f}" if sec_p_score is not None else "N/A")
               if 'image' in self.add_scores: line_parts.append(f"{i_score:.6f}" if i_score is not None else "N/A")
               line_parts.append(prompt)
               f.write("\t".join(line_parts) + "\n")

            # Worst prompts are no longer processed or saved to results.txt
            # if self.save_worst and worst_prompts:
            #     f.write("\nWORST PROMPTS:\n")
            #     f.write("\t".join(header_parts) + "\n")
            #     for i, (p_score, sec_p_score, prompt, i_score) in enumerate(zip(worst_scores, actual_worst_secondary_prompt_scores, worst_prompts, actual_worst_image_scores)):
            #         line_parts = [str(i+1)]
            #         if 'prompt' in self.add_scores:
            #             line_parts.append(f"{p_score:.6f}")
            #             if self.secondary_gt_prompt_score_for_model:
            #                 line_parts.append(f"{sec_p_score:.6f}" if sec_p_score is not None else "N/A")
            #         if 'image' in self.add_scores: line_parts.append(f"{i_score:.6f}" if i_score is not None else "N/A")
            #         line_parts.append(prompt)
            #         f.write("\t".join(line_parts) + "\n")

class LearnedEstimatorSaver(BaseSaver):
    """Saves the learned estimator theta vector to a file."""
    
    def save_result(self, results):
        # Get the list of thetas and corresponding model names
        thetas = results.get_thetas()
        model_names = results.metadata.get('scorer_model_names', [])

        if not thetas:
            print("Error: No theta vectors found in results for LearnedEstimatorSaver.")
            return

        if len(thetas) != len(model_names) and model_names: # Only warn if model_names were expected
            print(f"Warning: Mismatch between number of thetas ({len(thetas)}) and model names ({len(model_names)}). Filenames might be affected.")
        
        base_filename_param = self.params.get('filename', 'estimator.pt')
        base_name, base_ext = os.path.splitext(base_filename_param)

        for i, theta_to_save in enumerate(thetas):
            if theta_to_save is None:
                model_name_info = f"for model '{model_names[i]}'" if i < len(model_names) else f"at index {i}"
                print(f"Skipping save for estimator {model_name_info} as theta is None.")
                continue

            # Construct model-specific filename
            model_suffix = f"-{model_names[i]}" if i < len(model_names) and model_names[i] else f"-{i}"
            # Ensure model_suffix is not empty if model_names list was shorter than thetas list
            if not model_names and len(thetas) > 1: # Multiple thetas but no model names
                 model_specific_filename = f"{base_name}{model_suffix}{base_ext}"
            elif len(thetas) == 1 and not model_names: # Single theta, no model names (original behavior)
                 model_specific_filename = base_filename_param
            elif i < len(model_names) and model_names[i]: # Model name available
                 model_specific_filename = f"{base_name}-{model_names[i]}{base_ext}"
            else: # Fallback if model_names is short or entry is empty
                 model_specific_filename = f"{base_name}{model_suffix}{base_ext}"


            file_path = self.get_output_path(filename=model_specific_filename)
            
            if file_path is None: # Handle skip_existing
                print(f"Skipping save for {model_specific_filename} as it already exists and skip_existing is True.")
                continue
            
            try:
                torch.save(theta_to_save, file_path)
                print(f"Saved estimator theta (shape: {theta_to_save.shape}) to {file_path}")
            except Exception as e:
                print(f"Error saving estimator to {file_path}: {e}")

class VisitsSaver(BaseSaver):
    """Saves exploration visits to a file."""
    # Match BaseSaver's explicit arguments, remove **kwargs
    # Removed cfg argument
    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None,
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False,
                 **kwargs): # Add kwargs to accept unused arguments
        # Pass arguments explicitly to BaseSaver, including kwargs
        super().__init__(env=env, embedder=embedder, params=params,
                         scorer_model=scorer_model, results_dir=results_dir,
                         experiment_id=experiment_id, skip_existing=skip_existing,
                         **kwargs) # Pass kwargs to BaseSaver

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


class VisitsImageSaver(BaseSaver):
    """Saves images generated from visited trajectories for human feedback."""

    # Match BaseSaver's explicit arguments, remove **kwargs
    # Removed cfg argument, added specific config values needed
    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None,
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False,
                 # Core objects/config passed explicitly from LLMExperiment
                 seed: int = None,
                 total_repeats: int = 1,
                 algorithm: str = None,
                 **kwargs): # Accept kwargs for BaseSaver
        # Pass arguments explicitly to BaseSaver
        super().__init__(
            env=env,
            embedder=embedder,
            params=params,
            scorer_model=scorer_model,
            results_dir=results_dir,
            experiment_id=experiment_id,
            skip_existing=skip_existing,
            **kwargs # Pass unused args to BaseSaver
        )

        # --- Validate required objects (env, embedder should be set by BaseSaver) ---
        if self.env is None:
            # Add more context to the error
            raise ValueError("VisitsImageSaver requires the 'env' object. Ensure it's passed during instantiation and not overridden to null by config.")
        # Embedder is optional for BaseSaver but required here
        if self.embedder is None:
            raise ValueError("VisitsImageSaver requires the 'embedder' object. Ensure it's passed during instantiation.")
        # --- Derive/Store configuration ---
        self.horizon = self.env.max_episode_length # Derive horizon from env
        # Get dense_feedback and verbose from params (passed by Hydra)
        self.dense_feedback = self.params.get('dense_feedback', False) # Default to False if not in params
        self.verbose = self.params.get('verbose', False) # Default to False if not in params
        # Store explicitly passed seed, repeats, algorithm
        self.seed = seed
        self.total_repeats = total_repeats
        self.algorithm = algorithm
        # Get other params needed for image generation
        self.seed_per_prompt = self.params.get('seed_per_prompt', True)
        self.output_subdir = self.params.get('output_subdir', 'visit_images')
        # Allow overriding image generation settings for debugging/speed
        self.image_size = self.params.get('image_size', DEFAULT_CONFIG['image_size'])
        self.num_inference_steps = self.params.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])

        # --- Validate configuration ---
        if self.horizon is None or self.horizon <= 0:
             raise ValueError(f"VisitsImageSaver derived an invalid horizon ({self.horizon}) from env.")
        if self.seed is None:
            print("Warning: VisitsImageSaver initialized without a seed. Episode splitting will be disabled.")
        if self.total_repeats is None or self.total_repeats < 1:
            print(f"Warning: VisitsImageSaver initialized with invalid total_repeats ({self.total_repeats}). Defaulting to 1, episode splitting disabled.")
            self.total_repeats = 1
        if self.algorithm is None:
            raise ValueError("VisitsImageSaver requires the 'algorithm' name.")


        # --- Configuration for Image Generation (from params) ---
        self.seed_per_prompt = self.params.get('seed_per_prompt', True)
        self.output_subdir = self.params.get('output_subdir', 'visit_images')
        # image parameters are taken from DEFAULT_CONFIG when the generator is instantiated
 
    def save_result(self, results):
        """
        Generates and saves images based on visited trajectories.
        If dense_feedback is True in the config, generates images for each timestep h=1..H.
        Otherwise, generates images only for the full horizon H.
        """
        visits = results.visits
        if visits is None or not visits or not visits[0]:
            print("VisitsImageSaver: No visits data found in results. Skipping image generation.")
            return

        # Determine structure: visits[policy_idx][episode_idx] = (states, actions)
        try:
            num_policies = len(visits)
            num_episodes = len(visits[0])
            if num_policies == 0 or num_episodes == 0:
                 print("VisitsImageSaver: Visits data is empty. Skipping.")
                 return
            # Check structure of the first element
            first_visit = visits[0][0]
            if not isinstance(first_visit, tuple) or len(first_visit) != 2:
                 raise TypeError("Expected visits[p][e] to be a tuple (states, actions)")
            _ = first_visit[1] # Try accessing actions
        except (TypeError, IndexError, AttributeError) as e:
            print(f"VisitsImageSaver: Invalid visits structure: {e}. Skipping image generation.")
            print("Expected structure: List[List[Tuple[states, actions]]]")
            return

        # --- Use Horizon, Dense Feedback Flag, and Verbose Flag stored in self (from explicit __init__ args) ---
        horizon = self.horizon # Now from explicit arg
        dense_feedback = self.dense_feedback # Now from explicit arg
        verbose = self.verbose # Now from explicit arg
        print(f"VisitsImageSaver: Horizon={horizon}, Dense Feedback={dense_feedback}, Verbose={verbose}")

        # --- Setup Output Directory ---
        # We create a specific subdirectory for these images
        output_dir_path = os.path.join(self.results_dir, self.output_subdir)
        # Avoid adding experiment_id subdir if we are in the specific 'inspect' mode
        # Check against the default subdir name used in config_inspect.yaml
        if self.experiment_id and self.output_subdir != "visit_images_inspect":
            output_dir_path = os.path.join(output_dir_path, self.experiment_id)
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"VisitsImageSaver: Saving visit images to {output_dir_path}")

        # --- Initialize Image Generator ---
        # The generator will use defaults from image_generator.DEFAULT_CONFIG
        # for model_id, cache_dir, image_size, steps, guidance, seed etc.
        try:
            generator = StableDiffusionGenerator(
                num_inference_steps=self.num_inference_steps,
                image_size=self.image_size
            )
        except Exception as e:
            print(f"VisitsImageSaver: Failed to initialize StableDiffusionGenerator with defaults: {e}. Skipping.")
            return

        # --- Determine Timestep Range ---
        h_range = range(1, horizon + 1) if dense_feedback else range(horizon, horizon + 1)

        # --- Calculate Episode Range for this Seed ---
        start_ep_idx = 0
        end_ep_idx = num_episodes
        # Apply splitting only if seed and total_repeats are valid for distribution
        if self.seed is not None and self.total_repeats is not None and self.total_repeats > 1:
            # Ensure seed is 1-based for calculation
            current_seed_index = self.seed - 1 # Convert 1-based seed to 0-based index
            if current_seed_index < 0 or current_seed_index >= self.total_repeats:
                print(f"Warning: Invalid seed ({self.seed}) for total repeats ({self.total_repeats}). Processing all episodes.")
            else:
                # Calculate start and end indices using integer division for slicing
                start_ep_idx = current_seed_index * num_episodes // self.total_repeats
                end_ep_idx = (current_seed_index + 1) * num_episodes // self.total_repeats
                # Ensure end_ep_idx doesn't exceed num_episodes (shouldn't happen with this logic, but safe)
                end_ep_idx = min(end_ep_idx, num_episodes)
        
        # --- Log the episode range that will be processed ---
        # Use 1-based indexing for user-facing logs. The range is inclusive.
        # The loop `range(start_ep_idx, end_ep_idx)` will process episodes from start_ep_idx up to end_ep_idx - 1.
        # So the 1-based episode numbers are start_ep_idx + 1 to end_ep_idx.
        if end_ep_idx > start_ep_idx:
            if self.seed is not None and self.total_repeats is not None and self.total_repeats > 1:
                print(f"VisitsImageSaver (Seed {self.seed}/{self.total_repeats}): Processing episodes {start_ep_idx + 1} to {end_ep_idx} (inclusive) of {num_episodes} total episodes.", flush=True)
            else:
                print(f"VisitsImageSaver: Processing all episodes: {start_ep_idx + 1} to {end_ep_idx} (inclusive) of {num_episodes} total episodes.", flush=True)
        else:
            print(f"VisitsImageSaver: No episodes to process for this seed/range ({start_ep_idx+1} to {end_ep_idx}).", flush=True)

        # --- Generate and Save Images Per Episode and Timestep ---
        for ep_idx in range(start_ep_idx, end_ep_idx):
            # Add flush=True to ensure progress is visible
            # Log the absolute episode index (ep_idx + 1) relative to the total number of episodes
            print(f"VisitsImageSaver: Processing episode {ep_idx + 1}/{num_episodes}", flush=True)

            for h in h_range:
                # Add flush=True here too for timestep progress
                print(f"  Processing timestep h={h}/{horizon}", flush=True)
                timestep_images = []
                timestep_prompts = []

                for policy_idx in range(num_policies):
                    try:
                        # Extract actions for this policy and episode
                        # visits[policy_idx][ep_idx] should be (states, actions)
                        full_actions = visits[policy_idx][ep_idx][1]
                        if isinstance(full_actions, torch.Tensor):
                            full_actions = full_actions.cpu().numpy() # Ensure numpy array or list
                        full_actions = list(map(int, full_actions)) # Ensure list of ints

                        # --- Get Partial Actions for timestep h ---
                        partial_actions = full_actions[:h]

                        # Generate prompt from partial actions
                        prompt = create_prompt(partial_actions, self.env)
                        timestep_prompts.append(prompt)

                        # Set seed for this specific image generation if needed
                        if self.seed_per_prompt:
                                generator.seed = _get_seed_from_prompt(prompt)
                                generator.seed_generator()
                            # else: use the generator's current seed state (potentially incrementing)

                        # Generate image
                        # Adjust printing to show full prompt if short, or indicate if empty
                        prompt_display = prompt if prompt else "[Empty Prompt]"
                        if len(prompt_display) > 80:
                            prompt_display = prompt_display[:80] + "..."
                        print(f"    Generating image for policy {policy_idx + 1}/{num_policies} (Prompt: '{prompt_display}')")

                        # Pass the embedder instance to the sample method
                        # Ensure self.embedder is passed here
                        image_np, _ = generator.sample(prompt, embedder=self.embedder)
                        # Explicitly use PIL.Image to avoid potential name shadowing
                        timestep_images.append(PIL.Image.fromarray(image_np))

                    except IndexError:
                        print(f"    Warning: Missing visit data for episode {ep_idx}, policy {policy_idx}. Skipping.")
                        # Explicitly use PIL.Image. Use generator's image size (int) to create tuple.
                        placeholder_size = (generator.image_size, generator.image_size)
                        timestep_images.append(PIL.Image.new('RGB', placeholder_size, color = 'grey')) # Placeholder
                        timestep_prompts.append("Error: Missing Data")
                    except Exception as e:
                        print(f"    Error generating image for episode {ep_idx}, policy {policy_idx}, h={h}: {e}")
                        # Explicitly use PIL.Image. Use generator's image size (int) to create tuple.
                        placeholder_size = (generator.image_size, generator.image_size)
                        timestep_images.append(PIL.Image.new('RGB', placeholder_size, color = 'red')) # Error placeholder
                        timestep_prompts.append(f"Error: {e}")

                # --- Print Prompts if Verbose (using flag stored in self.verbose) ---
                if self.verbose: # Use self.verbose directly
                    print(f"    Timestep h={h} Prompts:")
                    for p_idx, p_text in enumerate(timestep_prompts):
                        print(f"      Policy {p_idx+1}: {p_text}")

                # --- Create and Save Grid Image for the Episode and Timestep ---
                if not timestep_images:
                    print(f"    No images generated for episode {ep_idx}, timestep {h}. Skipping grid.")
                    continue

                try:
                    n_cols = num_policies
                    n_rows = 1
                    # Create a compact figure with minimal margins and larger image area
                    fig, axes = plt.subplots(
                        n_rows, n_cols,
                        figsize=(5.5 * n_cols, 5.5 * n_rows),
                        squeeze=False
                    )

                    for i, img in enumerate(timestep_images):
                        ax = axes[0, i]
                        ax.imshow(img)
                        # Make policy title larger and do not show prompt text under images
                        ax.set_title(f"Policy {i+1}", fontsize=14)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Hide unused axes if any (shouldn't happen with n_rows=1)
                    for i in range(len(timestep_images), n_cols):
                        axes[0, i].axis('off')

                    # --- Determine the main title ---
                    # Using absolute episode index (ep_idx) and current timestep (h)
                    # Omit algorithm name for blindness in human feedback studies.
                    title_text = f"Episode: {ep_idx + 1}, Timestep: {h}"
                    
                    # Wrap the determined title text
                    wrapped_title = textwrap.fill(title_text, width=80) # Adjust width as needed
                    # Increase font size slightly, place near top
                    plt.suptitle(wrapped_title, fontsize=16, fontweight='bold', y=0.97)
                    # -----------------------------------------

                    # Tighten layout to reduce white background and enlarge images
                    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08, wspace=0.05, hspace=0.10)

                    # Construct filename including algorithm, episode, and timestep h
                    # Use the absolute episode index ep_idx in the filename
                    filename = f"alg-{self.algorithm}_episode_{ep_idx:03d}_timestep_{h:02d}.png"
                    output_path = os.path.join(output_dir_path, filename)

                    plt.savefig(output_path)
                    plt.close(fig)
                    print(f"    Saved grid image: {output_path}")

                except Exception as e:
                    print(f"    Error creating/saving grid image for episode {ep_idx}, timestep {h}: {e}")
                    # Ensure plot is closed even if saving fails
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                         plt.close(fig)


class ReadableVisitsSaver(BaseSaver):
    """
    Saves visit trajectories in a human-readable format (actions and prompts).
    Can load visits from a file specified in params['visits_path'] if not available
    in the results object.
    Respects dense_feedback setting to show prompts for prefixes.
    Saves output per policy.
    """
    # Updated __init__ to accept horizon and dense_feedback
    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None,
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False,
                 # Core objects/config passed explicitly from LLMExperiment
                 num_policies: int = 1, # Added num_policies argument
                 # (seed, total_repeats, algorithm are accepted by kwargs)
                 **kwargs): # Accept kwargs for BaseSaver
        # Pass common arguments to BaseSaver
        super().__init__(
            env=env,
            embedder=embedder,
            params=params,
            scorer_model=scorer_model,
            results_dir=results_dir,
            experiment_id=experiment_id,
            skip_existing=skip_existing,
            **kwargs # Pass unused args to BaseSaver
        )

        # --- Validate required objects (env should be set by BaseSaver) ---
        if self.env is None:
             raise ValueError(f"{self.__class__.__name__} requires the 'env' object.")
        # --- Derive/Store configuration ---
        self.horizon = self.env.max_episode_length # Derive horizon from env
        self.num_policies = num_policies # Store num_policies
        # Get dense_feedback from params (passed by Hydra)
        self.dense_feedback = self.params.get('dense_feedback', False) # Default to False if not in params

        # --- Validate configuration ---
        if self.horizon is None or self.horizon <= 0:
             raise ValueError(f"{self.__class__.__name__} derived an invalid horizon ({self.horizon}) from env.")

        print(f"Initialized {self.__class__.__name__} with params: {self.params}, "
              f"num_policies: {self.num_policies}, dense_feedback: {self.dense_feedback}, horizon: {self.horizon}")

    def save_result(self, results):
        print(f"Running {self.__class__.__name__}")

        raw_visits = results.visits
        visits_for_processing = []

        # Check if raw_visits are available
        if raw_visits is None:
            print(f"Error: No visits data available in results object for {self.__class__.__name__}. Skipping.")
            return
        if not isinstance(raw_visits, list):
            print(f"Error: Expected raw_visits to be a list, but got {type(raw_visits).__name__}. Skipping.")
            return
        if len(raw_visits) == 0:
            print(f"Error: Raw visits list is empty. Skipping.")
            return

        # Adapt structure based on self.num_policies
        if self.num_policies == 1:
            # Expected structure: List[Tuple(states, actions)]
            # We wrap it to be List[List[Tuple(states, actions)]] for uniform processing
            if not raw_visits: # Handles case where raw_visits = [[]] which is invalid for single policy
                print(f"Error: Raw visits for single policy is empty or malformed. Skipping.")
                return
            # Check if the first element is a tuple (indicative of single policy structure)
            if isinstance(raw_visits[0], tuple):
                visits_for_processing = [raw_visits]
            elif isinstance(raw_visits[0], list) and len(raw_visits) == 1 and isinstance(raw_visits[0][0], tuple):
                # This could be an already wrapped single policy, e.g. from a previous multi-policy run now treated as single
                visits_for_processing = raw_visits
            else:
                print(f"Error: Invalid visits structure for single policy. Expected List[Tuple(s,a)], got List[{type(raw_visits[0]).__name__}]. Skipping.")
                return
        else: # self.num_policies > 1
            # Expected structure: List[List[Tuple(states, actions)]]
            if not isinstance(raw_visits[0], list):
                print(f"Error: Invalid visits structure for multi-policy. Expected List[List[Tuple(s,a)]], got List[{type(raw_visits[0]).__name__}]. Skipping.")
                return
            visits_for_processing = raw_visits

        # Validate the standardized visits_for_processing structure
        if not visits_for_processing or not visits_for_processing[0]:
            print(f"Error: Visits data (after standardization) is empty or first policy has no visits. Skipping.")
            return
        try:
            first_visit_data = visits_for_processing[0][0] # Should now be List[List[Tuple(s,a)]]
            if not isinstance(first_visit_data, tuple) or len(first_visit_data) != 2:
                raise TypeError("Expected (states, actions) tuple in standardized visits structure")
            _ = first_visit_data[1] # Check actions access
        except (TypeError, IndexError) as e:
            print(f"Error: Invalid visit data structure after standardization: {e}. Expected List[List[Tuple(states, actions)]]. Skipping.")
            return

        # --- Process and Prepare Output ---
        output_lines = []
        try:
            # Determine structure from the standardized visits_for_processing
            num_policies_in_data = len(visits_for_processing)
            num_episodes = len(visits_for_processing[0]) # Assumes all policies have same num_episodes
            print(f"Processing {num_policies_in_data} policies and {num_episodes} episodes.")

            # Determine the range of horizons to generate prompts for
            horizon = self.horizon
            h_range = range(1, horizon + 1) if self.dense_feedback else range(horizon, horizon + 1)

            # Iterate through each policy and save to a separate file
            for p_idx in range(num_policies_in_data): # Use num_policies_in_data
                policy_output_lines = []
                policy_output_lines.append(f"--- Readable Visits: Policy {p_idx + 1} ---")
                policy_output_lines.append(f"Number of Episodes: {num_episodes}")
                policy_output_lines.append(f"Dense Feedback Mode: {self.dense_feedback}")
                policy_output_lines.append("-" * 25)

                for ep_idx in range(num_episodes):
                    policy_output_lines.append(f"\n  Episode {ep_idx + 1}:")
                    try:
                        # Use visits_for_processing for accessing data
                        full_actions = visits_for_processing[p_idx][ep_idx][1]
                        if isinstance(full_actions, torch.Tensor):
                            full_actions = full_actions.cpu().numpy()
                        full_actions = list(map(int, full_actions)) # Ensure list of ints

                        policy_output_lines.append(f"    Full Actions: {full_actions}")

                        # Generate prompts for relevant horizons (prefix lengths)
                        for h in h_range:
                            if h > len(full_actions): # Should not happen if horizon matches data
                                policy_output_lines.append(f"      h={h}: Error - Horizon exceeds action length")
                                continue

                            truncated_actions = full_actions[:h]
                            prompt = create_prompt(truncated_actions, self.env)
                            policy_output_lines.append(f"      h={h}:")
                            policy_output_lines.append(f"        Actions: {truncated_actions}")
                            policy_output_lines.append(f"        Prompt : '{prompt}'")

                    except IndexError:
                        policy_output_lines.append(f"    Error - Missing data for this episode")
                    except Exception as e:
                         policy_output_lines.append(f"    Error - Processing failed for this episode: {e}")

                # --- Save Policy-Specific File ---
                base_filename = self.params.get('output_filename', 'readable_visits.txt')
                name, ext = os.path.splitext(base_filename)
                policy_filename = f"{name}_policy_{p_idx+1}{ext}"
                output_path = self.get_output_path(filename=policy_filename)

                if output_path is None: # Skip if file exists and skip_existing is True
                     print(f"Skipping save for {policy_filename} as it already exists and skip_existing is True.")
                     continue

                # Print last few lines to console for confirmation - RESTORED
                print(f"\nPolicy {p_idx+1} - Last few lines:")
                print("\n".join(policy_output_lines[-5:]))

                try:
                    with open(output_path, 'w') as f:
                        f.write("\n".join(policy_output_lines))
                    print(f"Saved readable visits for policy {p_idx+1} to: {output_path}") # Restored individual save message
                except Exception as e:
                    print(f"Error saving readable visits for policy {p_idx+1} to {output_path}: {e}")
            
            # After the loop, print a summary message
            print(f"ReadableVisitsSaver: Finished processing {num_policies_in_data} policies. Files saved to '{self.results_dir}'.")

        except Exception as e:
            # Catch errors during the main processing loop (e.g., determining num_policies)
            print(f"Error processing visits: {e}")
            # Optionally save a general error file if needed
            error_filename = self.params.get('output_filename', 'readable_visits.txt') + ".error"
            error_path = self.get_output_path(filename=error_filename)
            if error_path:
                 try:
                     with open(error_path, 'w') as f:
                         f.write(f"Error during visit processing:\n{e}\n")
                         # Use visits_for_processing for error reporting
                         f.write(f"Processed visits type: {type(visits_for_processing)}\n")
                         if isinstance(visits_for_processing, tuple):
                             f.write(f"Tuple lengths: {[len(el) if hasattr(el, '__len__') else 'N/A' for el in visits_for_processing]}\n")
                 except Exception as e2:
                     print(f"Could not save error file: {e2}")

        # No return value needed for savers
