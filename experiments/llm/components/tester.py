import torch
import numpy as np
from abc import ABC, abstractmethod
from doexpy.env.llm import create_prompt_from_tokens

def generate_test_sequence(rng, word_lists, horizon):
    """Generate a test sequence using the appropriate word list for each horizon step"""
    return [
        " " if rng.random() < 0.1 else rng.choice(word_lists[i])
        for i in range(horizon)
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
        # Make sure testing_words_list is a list of lists with one list per horizon step
        if not isinstance(testing_words_list[0], list):
            testing_words_list = [testing_words_list] * horizon
            
        test_sequences = [generate_test_sequence(test_rng, testing_words_list, horizon) 
                for _ in range(N_test_prompts)]


        # Compute embeddings and predictions
        xtest = []
        ytest = []
        for sequence in test_sequences:
            # Create prompt directly from the sequence tokens
            prompt = create_prompt_from_tokens(sequence, env.base_prompt)
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

class ImageGenerationTester(BaseTester):
    def __init__(self, scorer_model, params=None):
        self.params = params or {}
        self.scorer_model = scorer_model
        self.take_best_worst_N = self.params.get('take_best_worst_N', 8) if self.params else 8
        super().__init__()
        
    def run_test(self, cfg, env, estimator, theta_star, training_words_list, testing_words_list):
        """Run the image generation test
        
        This tester finds the best and worst prompts based on the scorer model.
        """
        # Sample horizon-1 random tokens from testing set
        test_rng = np.random.RandomState(42)
        horizon = cfg.horizon
        prefix_length = horizon - 1
        
        # Make sure testing_words_list is a list of lists with one list per horizon step
        if not isinstance(testing_words_list[0], list):
            testing_words_list = [testing_words_list] * horizon
            
        # Generate a random prefix using the appropriate word list for each position
        prefix_sequence = [
            " " if test_rng.random() < 0.1 else test_rng.choice(testing_words_list[i])
            for i in range(prefix_length)
        ]
        
        # Score all possible completions
        all_scores = []
        all_prompts = []
        all_embeddings = []
        
        # Use the last horizon's word list for completions
        for last_token in testing_words_list[-1]:
            full_sequence = prefix_sequence + [last_token]
            prompt = create_prompt_from_tokens(full_sequence, env.base_prompt)
            score, embedding = self.scorer_model.score_prompt(prompt)
            
            all_scores.append(score.item())
            all_prompts.append(prompt)
            all_embeddings.append(embedding)
        
        # Sort by score (descending for best, ascending for worst)
        sorted_indices = np.argsort(all_scores)
        best_indices = sorted_indices[-self.take_best_worst_N:][::-1]  # Reverse to get descending order
        worst_indices = sorted_indices[:self.take_best_worst_N]
        
        # Get the best prompts and their scores
        best_prompts = [all_prompts[i] for i in best_indices]
        best_scores = [all_scores[i] for i in best_indices]
        
        # Get the worst prompts and their scores
        worst_prompts = [all_prompts[i] for i in worst_indices]
        worst_scores = [all_scores[i] for i in worst_indices]
        
        # Store the prompts and scores in the test results
        # Any saver can use this data if it knows how
        return {
            "image_generation": {
                "best_prompts": best_prompts,
                "best_scores": best_scores,
                "worst_prompts": worst_prompts,
                "worst_scores": worst_scores
            },
            "best_image_score": best_scores[0] if best_scores else 0,
            "worst_image_score": worst_scores[0] if worst_scores else 0,
            "avg_top_image_score": sum(best_scores) / len(best_scores) if best_scores else 0
        }

