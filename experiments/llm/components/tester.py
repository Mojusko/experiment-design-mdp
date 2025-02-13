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
    def __init__(self, scorer_model, params=None):
        self.params = params or {}
        self.scorer_model = scorer_model
        super().__init__()  # Call to parent if needed
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list):
        test_rng = np.random.RandomState(42)
        
        N_test_prompts = self.params['N_test_prompts']
        N_pairs_eval = self.params['N_pairs_eval']
        #if cfg.scorer_model in ['art','aesthetics']:
        #    N_test_prompts = 1000
        #    N_pairs_eval   = 5000
        #else:
        #    N_test_prompts = 50
        #    N_pairs_eval   = 100

        horizon = cfg.horizon
        test_sequences = [generate_test_sequence(test_rng, testing_words_list, horizon) 
                for _ in range(N_test_prompts)]

        # Compute embeddings and predictions
        xtest = []
        ytest = []
        for sequence in test_sequences:
           tokens = [t for t in sequence if t != " "]
           prompt = ", ".join(tokens) if env.base_prompt == '' else env.base_prompt + ", " + ", ".join(tokens)
           yy, feat = self.scorer_model.score_prompt(prompt)
           xtest.append(feat.detach().cpu())
           ytest.append(yy.detach().cpu())
        
        xtest = torch.vstack(xtest)
        ytest = torch.vstack(ytest)
        ypred = estimator.mean(xtest)

        pair_indices = np.array([(i, j) for i in range(N_test_prompts) for j in range(i+1, N_test_prompts)])
        selected_pairs = pair_indices[test_rng.choice(len(pair_indices), N_pairs_eval, replace=False)]
        
        correct_preferences = 0
        for i, j in selected_pairs:
            gt_prefers_i = (ytest[i] >= ytest[j]).item()
            pred_prefers_i = (ypred[i] >= ypred[j]).item()
            if gt_prefers_i == pred_prefers_i:
                correct_preferences += 1
        
        error = 1.0 - (correct_preferences / N_pairs_eval)


        return {"preference_error": error}

class CosineTester(BaseTester):
    def __init__(self, scorer_model, params=None):
        self.params = params or {}
        self.scorer_model = scorer_model
        super().__init__()
    
    @staticmethod
    def cosine_error(vec1, vec2): 
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
        # Compute cosine similarity and convert it to an error metric. 
        cos_sim = torch.nn.functional.cosine_similarity(vec1.flatten(), vec2.flatten(), dim=0)
        return 1 - cos_sim.item()
    
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list):
        # Get the ground-truth model weights and the estimated weights
        gt_weight = self.scorer_model.weight  # ground truth weight from aesthetics model
        est_weight = estimator.theta_fit      # estimated weight
        error = self.cosine_error(est_weight, gt_weight)
        return {"cosine_error": error}

