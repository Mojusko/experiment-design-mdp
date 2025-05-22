# Standard library imports
import json
import os
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

        self.take_best_worst_N = self.params.get('take_best_worst_N', 8)
        # Seed logic is specific here (_get_seed_from_prompt(self.base_prompt)), not using default directly
        self.debug_mode = self.params.get('debug_mode', False)
        # Get image_size and num_inference_steps from params or DEFAULT_CONFIG
        self.image_size = self.params.get('image_size', DEFAULT_CONFIG['image_size'])
        self.num_inference_steps = self.params.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
        self.base_prompt = self.params.get('base_prompt', '')  # Extract base_prompt, default to empty string
        self.add_image_score = self.params.get('add_image_score', False)  # Whether to add image scores
        self.metrics_filename = self.params.get('metrics_filename', 'image_metrics.json')
        self.save_worst = self.params.get('save_worst', False) # Add save_worst flag, default to False
        self.score_with_gt = self.params.get('score_with_gt', False) # New parameter, default to False
        
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
            best_scores = image_gen_data["best_scores"] # These are prompt scores from the tester
            worst_prompts = image_gen_data["worst_prompts"]
            worst_scores = image_gen_data["worst_scores"] # These are prompt scores from the tester
            
            # Generate images and potentially calculate new image scores
            self._generate_images(best_prompts, best_scores, worst_prompts, worst_scores, images_dir, results) # Pass results
            
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
            
            # Removed saving of best_scores_range and worst_scores_range as they are prompt scores
        
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
    
    def _generate_images(self, best_prompts, best_scores, worst_prompts, worst_scores, images_dir, results):
        """Generate images from lists of best and worst prompts (internal method)
        
        Args:
            best_prompts: List of best prompts to generate images for
            best_scores: List of scores for each best prompt (prompt scores from tester)
            worst_prompts: List of worst prompts to generate images for
            worst_scores: List of scores for each worst prompt (prompt scores from tester)
            images_dir: Directory to save images to
            results: ExperimentResults object, used to access estimator for image scoring
        """
        # Create experiment-specific subdirectory if experiment_id is provided
        if self.experiment_id:
            images_dir = os.path.join(images_dir, self.experiment_id)
            os.makedirs(images_dir, exist_ok=True)

        # Imports moved to top level

        # Initialize image generator using self attributes (derived from params/DEFAULT_CONFIG)
        # and specific seed logic for this saver.
        generator = StableDiffusionGenerator(
            stable_diffusion_id=DEFAULT_CONFIG['stable_diffusion_id'], # Use default model ID
            MODELS_CACHE_DIR=DEFAULT_CONFIG['MODELS_CACHE_DIR'], # Use default cache dir
            image_size=self.image_size, # Use size from __init__ (params or default)
            num_inference_steps=self.num_inference_steps, # Use steps from __init__ (params or default)
            guidance_scale=DEFAULT_CONFIG['guidance_scale'], # Use default guidance
            seed=_get_seed_from_prompt(self.base_prompt), # Specific seed logic for this saver
        )
        
        # Generate images for the best prompts
        best_generated_images = []
        # best_image_scores = [] # Removed unused variable

        # Print debug info
        if self.debug_mode:
            print(f"DEBUG MODE: Generating smaller images ({self.image_size}x{self.image_size}) with fewer steps ({self.num_inference_steps})")
        
        # Determine scoring model for images if add_image_score is True
        scoring_model_for_images = None
        if self.add_image_score:
            if self.score_with_gt:
                if self.scorer_model: # self.scorer_model is the GT model instance
                    scoring_model_for_images = self.scorer_model
                    print("ImageGenerationSaver: Will calculate image scores using the ground truth scorer model.")
                else:
                    print("Warning: ImageGenerationSaver.score_with_gt is true, but ground truth scorer_model is not available.")
            else: # Try to use learned estimator
                if results.estimators and results.estimators[0] and self.embedder:
                    try:
                        scoring_model_for_images = create_dot_product_model_from_estimator(results.estimators[0], self.embedder)
                        print("ImageGenerationSaver: Will calculate image scores using the first learned estimator.")
                    except Exception as e:
                        print(f"Warning: Could not create model from estimator for image scoring: {e}")
                else:
                    print("Warning: Cannot calculate image scores with estimator. First estimator or embedder not available, and score_with_gt is false.")

        actual_best_image_scores = []
        print("Generating images for BEST prompts:")
        for i, (full_prompt, prompt_score) in enumerate(zip(best_prompts, best_scores)): # Renamed score to prompt_score
            print(f"Generating best image {i+1}/{len(best_prompts)} for prompt: {full_prompt}")
            image, image_embedding = generator.sample(full_prompt, embedder=self.embedder)
            
            current_image_score = None
            if scoring_model_for_images:
                try:
                    # Ensure image_embedding is on the correct device for the model
                    img_score_tensor = scoring_model_for_images.score_embedding(image_embedding.to(self.embedder.device))
                    current_image_score = img_score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to score image for prompt '{full_prompt[:30]}...': {e}")
            actual_best_image_scores.append(current_image_score)

            filename_parts = [f"best_{i+1}"]
            if current_image_score is not None:
                filename_parts.append(f"image_{current_image_score:.4f}")
            img_filename = "_".join(filename_parts) + ".png"
            img_path = os.path.join(images_dir, img_filename)
            PIL.Image.fromarray(image).save(img_path)
            best_generated_images.append(image)
        
        worst_generated_images = [] # Initialize the list here
        actual_worst_image_scores = []
        print("\nGenerating images for WORST prompts:")
        for i, (full_prompt, prompt_score) in enumerate(zip(worst_prompts, worst_scores)): # Renamed score to prompt_score
            print(f"Generating worst image {i+1}/{len(worst_prompts)} for prompt: {full_prompt}")
            image, image_embedding = generator.sample(full_prompt, embedder=self.embedder)

            current_image_score = None
            if scoring_model_for_images:
                try:
                    img_score_tensor = scoring_model_for_images.score_embedding(image_embedding.to(self.embedder.device))
                    current_image_score = img_score_tensor.item()
                except Exception as e:
                    print(f"Warning: Failed to score image for prompt '{full_prompt[:30]}...': {e}")
            actual_worst_image_scores.append(current_image_score)

            filename_parts = [f"worst_{i+1}"]
            if current_image_score is not None:
                filename_parts.append(f"image_{current_image_score:.4f}")
            img_filename = "_".join(filename_parts) + ".png"
            img_path = os.path.join(images_dir, img_filename)
            PIL.Image.fromarray(image).save(img_path)
            worst_generated_images.append(image)

        # Create a summary image with generated images and their scores
        # Determine number of rows based on save_worst flag
        n_rows = 2 if self.save_worst and worst_generated_images else 1
        n_cols = len(best_prompts) # Use number of best prompts for columns
        if self.save_worst and worst_generated_images:
            n_cols = max(n_cols, len(worst_prompts)) # Adjust columns if worst are saved and longer

        if n_cols == 0:
             print("No images generated, skipping summary figure.")
             plt.close() # Ensure figure is closed if created implicitly
             return # Exit if no images to plot

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False) # Adjust height per row

        # Plot best images in the first row (axes[0, :])
        for i, (img, _prompt_score, full_prompt, img_score) in enumerate(zip(best_generated_images, best_scores, best_prompts, actual_best_image_scores)): # _prompt_score is unused
           ax = axes[0, i]
           ax.imshow(img)
           title = f"Best {i+1}"
           if img_score is not None:
               title += f" (Image Score: {img_score:.2f})"
           ax.set_title(title)
           wrapped_prompt = textwrap.fill(full_prompt, width=40)
           ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10)
           ax.set_xticks([])
           ax.set_yticks([])

        for i in range(len(best_generated_images), n_cols):
            axes[0, i].axis('off')

        if self.save_worst and worst_generated_images:
            for i, (img, _prompt_score, full_prompt, img_score) in enumerate(zip(worst_generated_images, worst_scores, worst_prompts, actual_worst_image_scores)): # _prompt_score is unused
               ax = axes[1, i]
               ax.imshow(img)
               title = f"Worst {i+1}"
               if img_score is not None:
                   title += f" (Image Score: {img_score:.2f})"
               ax.set_title(title)
               wrapped_prompt = textwrap.fill(full_prompt, width=40)
               ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10)
               ax.set_xticks([])
               ax.set_yticks([])

            for i in range(len(worst_generated_images), n_cols):
                axes[1, i].axis('off')
        elif n_rows == 2:
             for i in range(n_cols):
                  axes[1, i].axis('off')

        plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.5)
        summary_path = os.path.join(images_dir, "summary.png")
        plt.savefig(summary_path)
        plt.close()

        with open(os.path.join(images_dir, "results.txt"), "w") as f:
            f.write("BEST PROMPTS:\n")
            f.write("Rank\tImage Score\tPrompt\n") # Removed Prompt Score column
            for i, (_prompt_score, prompt, img_score) in enumerate(zip(best_scores, best_prompts, actual_best_image_scores)): # _prompt_score is unused
               img_score_str = f"{img_score:.6f}" if img_score is not None else "N/A"
               f.write(f"{i+1}\t{img_score_str}\t{prompt}\n") # Removed prompt_score

            if self.save_worst and worst_prompts:
                f.write("\nWORST PROMPTS:\n")
                f.write("Rank\tImage Score\tPrompt\n") # Removed Prompt Score column
                for i, (_prompt_score, prompt, img_score) in enumerate(zip(worst_scores, worst_prompts, actual_worst_image_scores)): # _prompt_score is unused
                    img_score_str = f"{img_score:.6f}" if img_score is not None else "N/A"
                    f.write(f"{i+1}\t{img_score_str}\t{prompt}\n") # Removed prompt_score

class LearnedEstimatorSaver(BaseSaver):
    """Saves the learned estimator theta vector to a file."""
    
    def save_result(self, results):
        # Note: This saver is skipped if num_scorer_models > 1 in LLMExperiment.test_and_save
        # If it runs, it means there's only one model/estimator.
        file_path = self.get_output_path(self.params.get('filename', 'estimator.pt'))
        if file_path is None: # Handle skip_existing
             print(f"Skipping save for {self.params.get('filename', 'estimator.pt')} as it already exists.")
             return

        # Get the list of thetas (should contain only one if this saver runs)
        thetas = results.get_thetas()

        # Check if the list is valid and contains at least one theta
        if thetas and thetas[0] is not None:
            theta_to_save = thetas[0] # Get the first (and only) theta
            try:
                # Use the correct variable theta_to_save
                torch.save(theta_to_save, file_path)
                print(f"Saved estimator theta (shape: {theta_to_save.shape}) to {file_path}")
            except Exception as e:
                print(f"Error saving estimator: {e}")
        else:
            # Update the error message for clarity
            print("Error: No valid theta vector found in results for LearnedEstimatorSaver.")

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
            generator = StableDiffusionGenerator()
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
                # Use inclusive start and exclusive end for clarity
                print(f"VisitsImageSaver (Seed {self.seed}/{self.total_repeats}): Processing episodes {start_ep_idx} (inclusive) to {end_ep_idx} (exclusive) (Total: {num_episodes})")
        else:
            # Use inclusive start and exclusive end for clarity
            print(f"VisitsImageSaver: Processing all episodes {start_ep_idx} (inclusive) to {end_ep_idx} (exclusive) (Seed/Repeats info not used for splitting).")

        # --- Generate and Save Images Per Episode and Timestep ---
        # Modify the loop to use the calculated range
        # --- Add explicit logging for the calculated range ---
        print(f"VisitsImageSaver: Calculated episode range for seed {self.seed}: {start_ep_idx} (inclusive) to {end_ep_idx} (exclusive)", flush=True)
        # ----------------------------------------------------
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
                    # Restore original figure size (e.g., width factor 5, height 8)
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 8 * n_rows), squeeze=False)

                    for i, (img, prompt) in enumerate(zip(timestep_images, timestep_prompts)):
                        ax = axes[0, i]
                        ax.imshow(img)
                        # Wrap prompt text using textwrap for better readability
                        wrapped_prompt = textwrap.fill(prompt, width=40) # Wrap at 40 characters
                        ax.set_title(f"Policy {i+1}", fontsize=10)
                        ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10) # Add padding
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Hide unused axes if any (shouldn't happen with n_rows=1)
                    for i in range(len(timestep_images), n_cols):
                        axes[0, i].axis('off')

                    # --- Determine the main title ---
                    title_prefix = "Base Prompt:"
                    base_prompt_content = ""
                    
                    # Try to get base_prompt from env first
                    if hasattr(self.env, 'base_prompt') and self.env.base_prompt:
                        base_prompt_content = self.env.base_prompt
                    else:
                        # Fall back to first policy's first action
                        try:
                            first_policy_actions = visits[0][ep_idx][1]
                            if isinstance(first_policy_actions, torch.Tensor):
                                first_policy_actions = first_policy_actions.cpu().numpy()
                            first_policy_actions = list(map(int, first_policy_actions))
                            
                            if first_policy_actions:
                                first_timestep_actions = first_policy_actions[:1]
                                base_prompt_content = create_prompt(first_timestep_actions, self.env)
                            else:
                                base_prompt_content = "[No actions for h=1]"
                        except Exception as e:
                            print(f"    Warning: Could not determine prompt for title: {e}")
                            base_prompt_content = f"Episode {ep_idx}"

                    # Wrap the determined title text
                    wrapped_title = textwrap.fill(f"{title_prefix} '{base_prompt_content}'", width=60) # Adjust width as needed
                    # Increase font size, make bold, lower position (adjust y value)
                    plt.suptitle(wrapped_title, fontsize=16, fontweight='bold', y=0.95)
                    # -----------------------------------------

                    # Adjust subplot parameters: increase bottom margin slightly to accommodate xlabels, adjust spacing
                    plt.subplots_adjust(bottom=0.25, hspace=0.4, wspace=0.2) # Increased bottom margin

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

                # Print last few lines to console for confirmation
                print(f"\nPolicy {p_idx+1} - Last few lines:")
                print("\n".join(policy_output_lines[-5:]))

                try:
                    with open(output_path, 'w') as f:
                        f.write("\n".join(policy_output_lines))
                    print(f"Saved readable visits for policy {p_idx+1} to: {output_path}")
                except Exception as e:
                    print(f"Error saving readable visits for policy {p_idx+1} to {output_path}: {e}")

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
