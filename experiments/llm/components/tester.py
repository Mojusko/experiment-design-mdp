import torch
import numpy as np
from abc import ABC, abstractmethod

def generate_test_sequence(rng, word_list, horizon):
    return [
        " " if rng.random() < 0.1 else rng.choice(word_list)
        for _ in range(horizon)
    ]

class BaseTester(ABC):
    @abstractmethod
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list):
        pass

class PreferenceTester(BaseTester):
    """
    Reproduces your final 'preference error' approach.
    """
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list):
        rng = np.random.RandomState(42)
        
        if cfg.scorer_model in ['art','aesthetics']:
            N_test_prompts = 1000
            N_pairs_eval   = 5000
        else:
            N_test_prompts = 50
            N_pairs_eval   = 100

        horizon = cfg.horizon
        test_sequences = [
            generate_test_sequence(rng, testing_words_list, horizon)
            for _ in range(N_test_prompts)
        ]

        xtest, ytest = [], []
        for seq in test_sequences:
            # If you want the same approach as "A plate with X"
            # or direct calls to theta_star(seq), etc.
            pass

        # Evaluate pairwise preferences, compute error
        # ...
        error = 0.123  # placeholder
        return {"preference_error": error}
