import os
import torch
import numpy as np

import hydra
from omegaconf import DictConfig

from stpy.helpers.helper import cartesian  # We'll use in _make_token_lists
from doexpy.env.llm import (
    LLMGrid, setup_clip_model, get_scorer_model, make_theta_star, CLIPEmbedder, generate_emissions
)
from components.feedback import FeedbackFactory
from components.solver  import SolverFactory
from components.tester  import BaseTester
from components.saver   import BaseSaver


def compute_prob_mae(emissions, estimator, scorer_model):
    """Compute mean absolute error between predicted and true pairwise probabilities."""
    # Get logits
    pred_logits = estimator.mean(emissions).to(emissions.device)
    true_logits = scorer_model.score_embedding(emissions)
    
    # Convert to probabilities
    pred_exp = torch.exp(pred_logits)
    true_exp = torch.exp(true_logits)
    
    # Compute pairwise probs efficiently
    pred_probs = pred_exp.view(-1, 1) / (pred_exp.view(-1, 1) + pred_exp.view(1, -1))
    true_probs = true_exp.view(-1, 1) / (true_exp.view(-1, 1) + true_exp.view(1, -1))
    
    # Get MAE from upper triangle
    mask = torch.triu(torch.ones_like(pred_probs), diagonal=1).bool()
    mae = torch.mean(torch.abs(pred_probs[mask] - true_probs[mask])).item()
    
    return mae

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
        
        os.makedirs(os.path.dirname(cfg.saver.params.path), exist_ok=True)
        if os.path.exists(cfg.saver.params.path):
            os.remove(cfg.saver.params.path)

        self.training_words, self.testing_words, self.model_words = self._load_data()
        self.env = self._init_env()
        self.feedback, self.design, self.estimator = FeedbackFactory.create(cfg, self.env)
        self.explorer = SolverFactory.create(cfg, self.env, self.design, self.feedback)
        self.tester = hydra.utils.instantiate(cfg.tester, scorer_model=self._scorer_model)
        self.saver = hydra.utils.instantiate(cfg.saver)
        
        self.visits = [] if self.cfg.feedback.num_policies == 1 else [[] for _ in range(self.cfg.feedback.num_policies )]
        
    def _perform_estimation(self, visits, update_design=False):
        """Only performs estimation using already collected data"""
        self.feedback.fit_estimator()
        mae = compute_prob_mae(self.env.emissions, self.estimator, self._scorer_model)
        print(f'T: {len(visits[0])}, MAE: {mae}', self.feedback.metrics)
        if update_design:
            self.design.update_estimator(self.estimator, self.env.emissions)

    def run(self):
        """
        Runs exploration with a single call to explorer.run, but uses a callback
        to buffer newly collected episodes. Once we've accumulated `freq` episodes
        (or hit the final episode), we refit to exactly that set of new visits.
    
        This ensures that the design always knows the total # of episodes,
        yet we only re-fit every `freq` episodes, similar to the old phased approach.
        """
        total_episodes = self.cfg.experiment.episodes
        freq = self.cfg.feedback.adaptive_estimation_frequency
        num_policies = self.cfg.feedback.num_policies
    
        # We'll keep a buffer of newly discovered episodes. For multi-policy, this is
        # a list of length num_policies, each an array of visits since the last re-fit.
        recent_visits_buffer = [[] for _ in range(num_policies)]
    
        def update_callback(ep_idx, new_visits_for_this_episode):
            # new_visits_for_this_episode is a list of length num_policies, each the
            # single newly collected episode for that policy.
    
            # Accumulate them in our buffer
            for policy_idx, single_visit in enumerate(new_visits_for_this_episode):
                recent_visits_buffer[policy_idx].append(single_visit)
    
            # Check if it's time to do a partial re-fit (freq episodes or end)
            if self.cfg.algorithm != 'random' and freq > 0 and (ep_idx + 1) % freq == 0:

                # Label just these newly collected episodes, then fit
                self.feedback.collect_labels(self.cfg, recent_visits_buffer, self._theta_star)
                self.feedback.fit_estimator()
                self.design.update_estimator(self.estimator, self.env.emissions)
    
                # Optionally measure partial MAE
                mae = compute_prob_mae(self.env.emissions, self.estimator, self._scorer_model)
                print(f"Episode {ep_idx+1} partial re-fit, MAE: {mae}", self.feedback.metrics)
    
                # Clear our buffer so next batch is fresh
                for p_i in range(num_policies):
                    recent_visits_buffer[p_i].clear()
    
        # Now call explorer.run exactly once, passing our callback
        # We request visitations so we can store them if desired
        # TODO: add support for numerical designs
        results = self.explorer.run(
            episodes=total_episodes,
            return_visitations=True,
            update_callback=update_callback
        )
        # results is typically (objective_values, opt, visits)
        # store them if needed
        *_, self.visits = results
    
    
        # Very end fit
        self.feedback.collect_labels(self.cfg, self.visits, self._theta_star)
        self.feedback.fit_estimator()
        self.design.update_estimator(self.estimator, self.env.emissions)
    
        # Finally, measure the MAE after the full run
        mae = compute_prob_mae(self.env.emissions, self.estimator, self._scorer_model)
        print(f"Final MAE after all episodes: {mae}")
        
    def _init_env(self):
        """
        Builds token lists (if 'optim' do cartesian, else repeated),
        sets up LLMGrid, plus a ground-truth scoring model.
        """
        # Setup CLIP
        self._clip_model, self._clip_processor, self._clip_tokenizer = setup_clip_model(self.cfg.cache_dir)

        # Generate separate emissions for ground truth model using model_words
        clip_embedder = CLIPEmbedder(self._clip_tokenizer, self._clip_model)
        model_emissions = generate_emissions(self.model_words, clip_embedder, self.cfg.cache_dir)

        # Create token_lists for training environment
        horizon = self.cfg.horizon
        if self.cfg.algorithm == "optim":
            combos = cartesian([self.training_words]*horizon)
            words_list = []
            for row in combos:
                tokens = [t for t in row if t != " "]
                words_list.append(", ".join(tokens))
            token_lists = [words_list]
        else:
            token_lists = [self.training_words]*horizon

        # Build environment
        env = LLMGrid(token_lists, self._clip_model, self._clip_processor, self._clip_tokenizer, self.cfg.cache_dir, base_prompt=self.cfg.base_prompt)

        # Build scorer using model emissions
        self._scorer_model = get_scorer_model(
            self.cfg.experiment.scorer_model,
            env.embedder,
            self._clip_model,
            self._clip_processor,
            self.cfg.cache_dir,
            model_emissions
        )
        self._theta_star = make_theta_star(env, self._scorer_model)
        return env

    def test_and_save(self):
        """Final estimation, testing and saving of results"""
        result_dict = self.tester.run_test(
            cfg=self.cfg,
            env=self.env,
            estimator=self.estimator,
            theta_star=self._theta_star,
            training_words_list=self.training_words,
            testing_words_list=self.testing_words
        )
        self.saver.save_result(result_dict)

    def _load_data(self):
        """Returns training_words, test_words, and model_words in 60-20-20 split"""
        # Get vocabulary file(s) from config
        vocab_files = self.cfg.experiment.vocabulary
        if isinstance(vocab_files, str):
            vocab_files = [vocab_files]
        rng = np.random.RandomState(42)
    
        # Load and deduplicate vocabulary
        full_list = []
        for path in vocab_files:
            with open(path, 'r') as f:
                full_list.extend([line.strip() for line in f])
        full_list = list(dict.fromkeys(full_list))
    
        # Cap vocabulary if needed
        if len(full_list) > self.cfg.experiment.vocab_size:
            print(f"Capping data at {self.cfg.experiment.vocab_size} items")
            full_list = list(rng.choice(full_list, self.cfg.experiment.vocab_size, replace=False))
    
        # Create 60-20-20 split
        n_total = len(full_list)
        n_train = int(0.6 * n_total)
        n_test = int(0.2 * n_total)
        
        indices = rng.permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:n_train + n_test]
        model_idx = indices[n_train + n_test:]
    
        training_words = [full_list[i] for i in train_idx]
        testing_words = [full_list[i] for i in test_idx]
        model_words = [full_list[i] for i in model_idx]
    
        return training_words, testing_words, model_words
