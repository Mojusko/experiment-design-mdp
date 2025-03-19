import os
import torch
import numpy as np
import datetime

import hydra
from omegaconf import DictConfig

from stpy.helpers.helper import cartesian  # We'll use in _make_token_lists
from doexpy.env.llm import (
    LLMGrid, setup_clip_model, get_scorer_model, make_theta_star, CLIPEmbedder, generate_emissions
)
from components.feedback import FeedbackFactory
from components.solver  import SolverFactory
from components.tester  import BaseTester, ImageGenerationTester
from components.saver   import BaseSaver



class LLMExperiment:
    """
    High-level orchestrator. Steps:
      1) Load data & environment
      2) Initialize feedback+design+estimator
      3) Initialize solver (FrankWolfe, DP, random, etc.) -> Explorer
      4) Run exploration
      5) Fit/estimate
      6) Test & save results
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(int(cfg.seed))
        
        # Create results directory with timestamp
        timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.results_dir = f"{cfg.results_dir}-{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        #self.training_words, self.testing_words, self.model_words = self._load_data_legacy()
        self.training_words, self.testing_words, self.model_words = self._load_data()
        self.env = self._init_env()
        self.env._scorer_vector = self._scorer_model.weight
        self.feedback, self.design, self.estimator = FeedbackFactory.create(cfg, self.env)
        self.explorer = SolverFactory.create(cfg, self.env, self.design, self.feedback)
        
        # Build experiment_id with prefix if available
        experiment_id = str(self.cfg.experiment_id) if self.cfg.experiment_id is not None else ""
        if hasattr(self.cfg.experiment, 'id_prefix') and self.cfg.experiment.id_prefix:
            # Check if prefix is already present
            prefix_present = experiment_id.startswith(self.cfg.experiment.id_prefix) if experiment_id else False
            
            if not prefix_present:
                # Only prepend if not already present
                algorithm_code = self._get_algorithm_code()
                feedback_code = self._get_feedback_code()
                if experiment_id:
                    # If experiment_id is already set, use it as a suffix (typically seed number)
                    experiment_id = f"{self.cfg.experiment.id_prefix}-{algorithm_code}-{feedback_code}-{experiment_id}"
                else:
                    experiment_id = f"{self.cfg.experiment.id_prefix}-{algorithm_code}-{feedback_code}"
        
        self.experiment_id = experiment_id
        
        # Initialize testers and savers with results_dir and experiment_id
        self.testers = [hydra.utils.instantiate(t, scorer_model=self._scorer_model) for t in self.cfg.tester]
        
        # Initialize savers with just what they need
        self.savers = [
            hydra.utils.instantiate(
                s, 
                scorer_model=self._scorer_model,
                results_dir=self.results_dir,
                experiment_id=self.experiment_id
            ) for s in self.cfg.savers
        ]
        self.visits = [] if self.cfg.feedback.num_policies == 1 else [[] for _ in range(self.cfg.feedback.num_policies)]

    def run(self):
        total_episodes = self.cfg.experiment.episodes
        est_freq = self.cfg.feedback.adaptive_estimation_frequency
        est_start = self.cfg.feedback.adaptive_estimation_start
        num_policies = self.cfg.feedback.num_policies

        all_visits = [[] for _ in range(num_policies)]
        recent_visits_buffer = [[] for _ in range(num_policies)]

        def update_callback(ep_idx, new_visits_for_this_episode):
            for policy_idx, single_visit in enumerate(new_visits_for_this_episode):
                all_visits[policy_idx].append(single_visit)
                recent_visits_buffer[policy_idx].append(single_visit)

            if est_freq > 0 and ep_idx < total_episodes - 1 and ep_idx >= est_start and ep_idx % est_freq == 0:
                self.feedback.collect_labels(self.cfg, recent_visits_buffer, self._theta_star)
                self.feedback.fit_estimator()
                self.design.update_estimator(self.estimator, self.env.emissions)
                print(f"Episode {ep_idx} partial re-fit complete")
                for p_i in range(num_policies):
                    recent_visits_buffer[p_i].clear()

        results = self.explorer.run(
            episodes=total_episodes,
            return_visitations=True,
            update_callback=update_callback
        )
        self.visits = results

        if any(len(buf) > 0 for buf in recent_visits_buffer):
            self.feedback.collect_labels(self.cfg, recent_visits_buffer, self._theta_star)
        elif est_start == 0:
            self.feedback.collect_labels(self.cfg, all_visits, self._theta_star)

        self.feedback.fit_estimator()
        self.design.update_estimator(self.estimator, self.env.emissions)
        print(f"Final estimation after all {total_episodes} episodes complete")
        
    def _init_env(self):
        """
        Builds token lists (if 'optim' do cartesian, else repeated),
        sets up LLMGrid, plus a ground-truth scoring model.
        """
        # Setup CLIP
        self._clip_model, self._clip_processor, self._clip_tokenizer = setup_clip_model(self.cfg.cache_dir)

        # Generate separate emissions for ground truth model using model_words
        clip_embedder = CLIPEmbedder(self._clip_tokenizer, self._clip_model)
        #model_emissions = generate_emissions(self.model_words, clip_embedder, self.cfg.cache_dir) if self.model_words else None

        # Create token_lists for training environment
        horizon = self.cfg.horizon
        if self.cfg.algorithm == "optim":
            # For optim, we need to create all possible combinations
            combos = cartesian(self.training_words)
            words_list = []
            for row in combos:
                tokens = [t for t in row if t != " "]
                words_list.append(", ".join(tokens))
            token_lists = [words_list]
        else:
            # Use the horizon-specific token lists
            token_lists = self.training_words

        # Build environment
        env = LLMGrid(token_lists, self._clip_model, self._clip_processor, self._clip_tokenizer, self.cfg.cache_dir, self.cfg.normalize_CLIP, base_prompt=self.cfg.base_prompt, include_base_prompt_in_first_tokens=self.cfg.include_base_prompt_in_first_tokens, verbose=self.cfg.verbose)

        # Build scorer using model emissions with non-normalized embedder
        self._scorer_model = get_scorer_model(
            self.cfg.experiment.scorer_model,
            env,
            self._clip_model,
            self._clip_processor,
            self.cfg.cache_dir,
        )
        self._theta_star = make_theta_star(env, self._scorer_model)
        return env

    def test_and_save(self):
        """Final estimation, testing and saving of results"""
        # Create a container for all results
        from components.results import ExperimentResults
        results = ExperimentResults()
        
        # Set the estimator and visits
        results.set_estimator(self.estimator)
        results.set_visits(self.visits)
        
        # Add experiment metadata
        results.add_metadata('horizon', self.cfg.horizon)
        results.add_metadata('algorithm', self.cfg.algorithm)
        results.add_metadata('base_prompt', self.cfg.base_prompt)
        
        # Run all testers and collect metrics
        for tester in self.testers:
            tester_results = tester.run_test(
                cfg=self.cfg,
                env=self.env,
                estimator=self.estimator,
                theta_star=self._theta_star,
                training_words_list=self.training_words,
                testing_words_list=self.testing_words
            )
            
            # Add metrics to results container
            results.add_metrics(tester_results)
        
        # Use all savers to save the results
        for saver in self.savers:
            saver.save_result(results)

    def _get_algorithm_code(self):
        """Get a short code for the algorithm type"""
        algorithm = self.cfg.algorithm.lower()
        if algorithm == "design":
            return "dsn"
        elif algorithm == "random":
            return "rand"
        elif algorithm == "optim":
            return "opt"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Expected one of: design, random, optim")
            
    def _get_feedback_code(self):
        """Get a short code for the feedback type"""
        feedback = self.cfg.feedback.name.lower()
        if feedback == "numerical":
            return "num"
        elif feedback == "multinomial":
            return "mult"
        else:
            return feedback[:3]  # First 3 chars as fallback
            
    def _load_data(self):
        """Returns training_words_lists and testing_words_lists for each horizon step"""
        vocab_files = self.cfg.experiment.vocabulary
        
        # Check if horizon matches the number of vocabulary files
        # Make sure vocab_files is a flat list, not a list of lists
        if len(vocab_files) != self.cfg.horizon:
            raise ValueError(f"Number of vocabulary files ({len(vocab_files)}) must match horizon ({self.cfg.horizon})")
        
        rng = np.random.RandomState(42)
        
        training_words_lists = []
        testing_words_lists = []
        
        # Process each vocabulary file separately
        for path in vocab_files:
            # Load words from file
            with open(path, 'r') as f:
                full_list = [line.strip() for line in f]
            full_list = list(dict.fromkeys(full_list))  # Remove duplicates
            
            # Cap vocabulary if needed
            if len(full_list) > self.cfg.experiment.vocab_size:
                print(f"Capping data from {path} at {self.cfg.experiment.vocab_size} items")
                full_list = list(rng.choice(full_list, self.cfg.experiment.vocab_size, replace=False))
            
            # Create 75-25 split for this file
            n_total = len(full_list)
            n_train = int(0.75 * n_total)
            
            indices = rng.permutation(n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            # Add to lists
            training_words_lists.append([full_list[i] for i in train_idx])
            testing_words_lists.append([full_list[i] for i in test_idx])
        
        return training_words_lists, testing_words_lists, []
