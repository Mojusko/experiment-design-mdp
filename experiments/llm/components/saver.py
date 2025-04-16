import numpy as np
import json
import os
import torch
import matplotlib.pyplot as plt
# Import PIL module only (no direct Image import)
import PIL
import yaml
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path # Import Hydra path utility
# Import necessary components for VisitsImageSaver at the top level
from doexpy.env.llm import create_prompt
from experiments.llm.image_generator import StableDiffusionGenerator, _get_seed_from_prompt, DEFAULT_CONFIG
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

    # Use only named arguments, remove **kwargs to be explicit
    # Removed cfg argument
    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None, # Hydra populates this from config
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False):
        # Store core objects and config if provided
        # Dummy args (horizon, dense_feedback, verbose) removed - Python ignores extra args passed during instantiation
        self.env = env
        self.embedder = embedder
        # self.cfg is removed
        self.params = params if params is not None else {} # Use the passed params DictConfig
        self.scorer_model = scorer_model
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.skip_existing = skip_existing

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
        # Import DEFAULT_CONFIG here if needed for defaults, or rely on _generate_images
        from experiments.llm.image_generator import DEFAULT_CONFIG

        self.take_best_worst_N = self.params.get('take_best_worst_N', 8)
        # Seed logic is specific here (_get_seed_from_prompt(base_prompt)), not using default directly
        self.debug_mode = self.params.get('debug_mode', False)
        # Get image_size and num_inference_steps from params or DEFAULT_CONFIG
        self.image_size = self.params.get('image_size', DEFAULT_CONFIG['image_size'])
        self.num_inference_steps = self.params.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
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

        # Import generator classes, seed function, and DEFAULT_CONFIG locally
        from experiments.llm.image_generator import StableDiffusionGenerator, _get_seed_from_prompt, DEFAULT_CONFIG

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
            PIL.Image.fromarray(image).save(img_path)
            
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
            PIL.Image.fromarray(image).save(img_path)
            
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
    # Match BaseSaver's explicit arguments, remove **kwargs
    # Removed cfg argument
    def __init__(self,
                 env=None,
                 embedder=None,
                 params: DictConfig = None,
                 scorer_model=None,
                 results_dir=None,
                 experiment_id=None,
                 skip_existing: bool = False):
        # Pass arguments explicitly to BaseSaver (without cfg)
        # Dummy args (horizon, dense_feedback, verbose) removed - Python ignores extra args passed during instantiation
        super().__init__(env=env, embedder=embedder, params=params,
                         scorer_model=scorer_model, results_dir=results_dir,
                         experiment_id=experiment_id, skip_existing=skip_existing)

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
                 # Specific config values passed from LLMExperiment
                 horizon: int = None,
                 dense_feedback: bool = False,
                 verbose: bool = False):
        # Pass arguments explicitly to BaseSaver (without cfg)
        # BaseSaver no longer has dummy args for horizon, dense_feedback, verbose
        super().__init__(env=env, embedder=embedder, params=params,
                         scorer_model=scorer_model, results_dir=results_dir,
                         experiment_id=experiment_id, skip_existing=skip_existing)

        # Store the specific config values needed by this saver
        self.horizon = horizon
        self.dense_feedback = dense_feedback
        self.verbose = verbose

        # --- Validate required objects (self.env should now be set correctly by BaseSaver) ---
        if self.env is None:
            # Add more context to the error
            raise ValueError("VisitsImageSaver requires the 'env' object. Ensure it's passed during instantiation and not overridden to null by config.")
        if self.embedder is None:
            raise ValueError("VisitsImageSaver requires the 'embedder' object. Ensure it's passed during instantiation.")
        # Removed cfg check as it's no longer passed/stored
        # if self.cfg is None:
        #      raise ValueError("VisitsImageSaver requires the 'cfg' object. Ensure it's passed during instantiation.")
        if self.horizon is None: # Add check for horizon as it's critical
             raise ValueError("VisitsImageSaver requires the 'horizon' value. Ensure it's passed during instantiation.")


        # --- Configuration for Image Generation (using DEFAULT_CONFIG and self.params) ---
        # self.params is now directly passed and stored by BaseSaver
        self.image_size = self.params.get('image_size', DEFAULT_CONFIG['image_size'])
        self.num_inference_steps = self.params.get('num_inference_steps', DEFAULT_CONFIG['num_inference_steps'])
        self.guidance_scale = self.params.get('guidance_scale', DEFAULT_CONFIG['guidance_scale'])
        # Default to using prompt-specific seeds for reproducibility per prompt
        self.seed_per_prompt = self.params.get('seed_per_prompt', True)
        self.base_seed = self.params.get('seed', DEFAULT_CONFIG['seed']) # Base seed if not using seed_per_prompt
        self.output_subdir = self.params.get('output_subdir', 'visit_images') # Specific to this saver
        self.stable_diffusion_id = self.params.get('stable_diffusion_id', DEFAULT_CONFIG['stable_diffusion_id'])
        self.models_cache_dir = self.params.get('models_cache_dir', DEFAULT_CONFIG['MODELS_CACHE_DIR'])

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

        # --- Use Horizon, Dense Feedback Flag, and Verbose Flag stored in self ---
        # These values are guaranteed to exist due to checks in __init__
        horizon = self.horizon
        dense_feedback = self.dense_feedback
        verbose = self.verbose
        print(f"VisitsImageSaver: Horizon={horizon}, Dense Feedback={dense_feedback}, Verbose={verbose}")


        # --- Setup Output Directory ---
        # We create a specific subdirectory for these images
        output_dir_path = os.path.join(self.results_dir, self.output_subdir)
        if self.experiment_id:
            output_dir_path = os.path.join(output_dir_path, self.experiment_id)
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"VisitsImageSaver: Saving visit images to {output_dir_path}")

        # --- Initialize Image Generator ---
        try:
            generator = StableDiffusionGenerator(
                stable_diffusion_id=self.stable_diffusion_id,
                MODELS_CACHE_DIR=self.models_cache_dir,
                image_size=self.image_size,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.base_seed # Initial seed
            )
        except Exception as e:
            print(f"VisitsImageSaver: Failed to initialize StableDiffusionGenerator: {e}. Skipping.")
            return

        # --- Determine Timestep Range ---
        h_range = range(1, horizon + 1) if dense_feedback else range(horizon, horizon + 1)

        # --- Generate and Save Images Per Episode and Timestep ---
        for ep_idx in range(num_episodes):
            print(f"VisitsImageSaver: Processing episode {ep_idx + 1}/{num_episodes}")

            for h in h_range:
                print(f"  Processing timestep h={h}/{horizon}")
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
                        image_np, _ = generator.sample(prompt, embedder=self.embedder)
                        # Explicitly use PIL.Image to avoid potential name shadowing
                        timestep_images.append(PIL.Image.fromarray(image_np))

                    except IndexError:
                        print(f"    Warning: Missing visit data for episode {ep_idx}, policy {policy_idx}. Skipping.")
                        # Explicitly use PIL.Image
                        timestep_images.append(PIL.Image.new('RGB', (self.image_size, self.image_size), color = 'grey')) # Placeholder
                        timestep_prompts.append("Error: Missing Data")
                    except Exception as e:
                        print(f"    Error generating image for episode {ep_idx}, policy {policy_idx}, h={h}: {e}")
                        # Explicitly use PIL.Image
                        timestep_images.append(PIL.Image.new('RGB', (self.image_size, self.image_size), color = 'red')) # Error placeholder
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
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)

                    for i, (img, prompt) in enumerate(zip(timestep_images, timestep_prompts)):
                        ax = axes[0, i]
                        ax.imshow(img)
                        # Wrap prompt text for display below the image
                        wrapped_prompt = '\n'.join(prompt[j:j+60] for j in range(0, len(prompt), 60)) # Adjust wrap length if needed
                        ax.set_title(f"Policy {i+1}", fontsize=10)
                        ax.set_xlabel(wrapped_prompt, fontsize=8, labelpad=10) # Add padding
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Hide unused axes if any (shouldn't happen with n_rows=1)
                    for i in range(len(timestep_images), n_cols):
                        axes[0, i].axis('off')

                    plt.suptitle(f"Episode {ep_idx} - Timestep {h}", fontsize=14)
                    # Adjust subplot parameters for more bottom space for x-labels (prompts)
                    plt.subplots_adjust(bottom=0.2, hspace=0.3) # Increase bottom margin and horizontal space

                    # Construct filename including timestep h
                    filename = f"episode_{ep_idx:03d}_timestep_{h:02d}.png"
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
                 # Specific config values passed from LLMExperiment
                 horizon: int = None,
                 dense_feedback: bool = False):
        # Pass common arguments to BaseSaver
        super().__init__(env=env, embedder=embedder, params=params,
                         scorer_model=scorer_model, results_dir=results_dir,
                         experiment_id=experiment_id, skip_existing=skip_existing)

        # Store specific config values needed by this saver
        self.horizon = horizon
        self.dense_feedback = dense_feedback

        # --- Validate required objects and config ---
        if self.env is None:
             raise ValueError(f"{self.__class__.__name__} requires the 'env' object.")
        if self.horizon is None:
             raise ValueError(f"{self.__class__.__name__} requires the 'horizon' value.")

        print(f"Initialized {self.__class__.__name__} with params: {self.params}, "
              f"horizon: {self.horizon}, dense_feedback: {self.dense_feedback}")

    def save_result(self, results):
        print(f"Running {self.__class__.__name__}")

        loaded_visits = None
        # Prioritize visits from the results object
        if results.visits and results.visits[0]: # Check if visits exist and are not empty
            print("Using visits provided by the experiment results.")
            loaded_visits = results.visits
        # If not available in results, try loading from path specified in params
        elif self.params.get('visits_path'):
            visits_path = self.params['visits_path']
            # Resolve to absolute path before checking existence
            absolute_visits_path = to_absolute_path(visits_path) if visits_path else None
            print(f"Attempting to load visits from absolute path in params: {absolute_visits_path}")

            if absolute_visits_path and os.path.exists(absolute_visits_path):
                try:
                    loaded_visits = torch.load(absolute_visits_path)
                    print(f"Successfully loaded visits from {absolute_visits_path}")
                except Exception as e:
                    print(f"Error loading visits from {absolute_visits_path}: {e}. Skipping saver.")
                    return # Stop execution for this saver
            elif visits_path:
                # Print the absolute path tried
                print(f"Warning: Visits path specified in params but not found: {absolute_visits_path}. Skipping saver.")
                return
            else:
                 print("Warning: visits_path specified in params is null or empty. Skipping saver.")
                 return
        else:
            print("Warning: No visits available in results and no visits_path specified in params. Skipping saver.")
            return

        # Assume loaded_visits is the correct list structure [policy][episode](states, actions)
        # Remove the check for the old 3-tuple format.

        # Check if the loaded visits list is empty or if its first element is empty
        # Use explicit checks instead of relying on truthiness of arrays/lists
        if loaded_visits is None or len(loaded_visits) == 0 or len(loaded_visits[0]) == 0:
              print("Error: Visits data is empty or failed to load. Skipping saver.")
              return

        # --- Process and Prepare Output ---
        output_lines = []
        try:
            # Determine structure: visits[policy_idx][episode_idx] = (states, actions)
            # Use loaded_visits directly for processing
            num_policies = len(loaded_visits)
            num_episodes = len(loaded_visits[0])
            print(f"Processing {num_policies} policies and {num_episodes} episodes.")

            # Determine the range of horizons to generate prompts for
            h_range = range(1, self.horizon + 1) if self.dense_feedback else range(self.horizon, self.horizon + 1)

            # Iterate through each policy and save to a separate file
            for p_idx in range(num_policies):
                policy_output_lines = []
                policy_output_lines.append(f"--- Readable Visits: Policy {p_idx + 1} ---")
                policy_output_lines.append(f"Number of Episodes: {num_episodes}")
                policy_output_lines.append(f"Dense Feedback Mode: {self.dense_feedback}")
                policy_output_lines.append("-" * 25)

                for ep_idx in range(num_episodes):
                    policy_output_lines.append(f"\n  Episode {ep_idx + 1}:")
                    try:
                        # visits[policy_idx][ep_idx] = (states, actions)
                        # Use loaded_visits directly for accessing data
                        full_actions = loaded_visits[p_idx][ep_idx][1]
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
                         f.write(f"Loaded visits type: {type(loaded_visits)}\n")
                         if isinstance(loaded_visits, tuple):
                             f.write(f"Tuple lengths: {[len(el) if hasattr(el, '__len__') else 'N/A' for el in loaded_visits]}\n")
                 except Exception as e2:
                     print(f"Could not save error file: {e2}")

        # No return value needed for savers
