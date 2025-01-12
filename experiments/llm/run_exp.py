import torch
import os
import argparse
from stpy.helpers.helper import cartesian
from functools import partial
from typing import List, Tuple, Optional, Union
from PIL import Image
import time
from image_generator import StableDiffusionGenerator

from doexpy.env.llm import LLMGrid
from doexpy.functionals.doe_static_functionals import DesignA, DesignD, MultiPolicyAggDesignA, MultiPolicyAggDesignD, MultiPolicyOrigDesignD
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

from stpy.regression.kernelized_features import KernelizedFeatures
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
from stpy.probability.multinomial_likelihood import MultinomialLikelihood
from stpy.regularization.regularizer import L2Regularizer
from stpy.embeddings.polynomial_embedding import CustomEmbedding
import torch.nn as nn
import numpy as np

class CLIPEmbedder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def embed_text(self, text: str, normalize: bool = False) -> torch.Tensor:
        """Embed text using CLIP model
        
        Args:
            text: Text to embed
            normalize: Whether to L2 normalize the embedding
            
        Returns:
            Text embedding
        """
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embedding = self.model.get_text_features(**text_input).detach().double()
        
        if normalize:
            embedding = embedding / torch.norm(embedding, p=2)
            
        return embedding.view(1, -1)

class DotProductModel(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
    
    def forward(self, x):
        return torch.mm(x, self.embedding.T)

def create_prompt(actions: List[int], env, prefix: str = 'A plate with ') -> str:
    """Create prompt from action sequence"""
    tokens = [env.unique_elements[int(action)] for action in actions]
    valid_tokens = [t for t in tokens if t != " "]
    return prefix + ", ".join(valid_tokens) if valid_tokens else prefix.rstrip()

def setup_clip_model(env) -> Tuple[CLIPEmbedder, DotProductModel]:
    """Initialize CLIP embedder and dot product model"""
    embedder = CLIPEmbedder(env._tokenizer, env._model)
    art_embedding = embedder.embed_text('art', normalize=True)
    model = DotProductModel(art_embedding).eval()
    return embedder, model

def make_theta_star(env, embedder, model):
    """Create theta_star function with environment and models bound"""
    def theta_star(actions: List[int], returnx: bool = False, verbose: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute similarity score between action sequence and art concept"""
        assert len(actions) > 0
        
        prompt = create_prompt(actions, env)
        if verbose:
            print(prompt)

        feat = embedder.embed_text(prompt)
        val = model(feat)
        
        return (val, feat) if returnx else val
    
    return theta_star

def generate_test_sequence(test_rng, testing_words_list: List[str], horizon: int) -> List[str]:
    """Generate a single test sequence"""
    return [" " if test_rng.random() < 0.1 else test_rng.choice(testing_words_list) 
            for _ in range(horizon)]

# Parse arguments
parser = argparse.ArgumentParser(description='LLM experiment.')
parser.add_argument('--episodes', default=10, type=int, help='Episodes')
parser.add_argument('--feedback_type', default='multinomial', type=str, choices=['multinomial', 'numerical'], help='Type of feedback')
parser.add_argument('--algorithm', default='greedy', type=str, help='type of algorithm')
parser.add_argument('--num_components', default=100, type=int, help='Number of components in FW')
parser.add_argument('--save', default="results/experiment.csv", type=str, help='name of the file')
parser.add_argument('--seed', default=12, type=str, help='Use this to set the seed for the random number generator')
parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
parser.add_argument('--opt', default=None, type=str, help='whether to return opt')
parser.add_argument('--lambda_reg', default=1.0, type=float, help='Regularization parameter lambda')
parser.add_argument('--dense_feedback', action='store_true', help='Use dense feedback along trajectory')
parser.add_argument('--cache_dir', default=os.path.expanduser('~/.cache/huggingface/hub'), 
                    type=str, help='Model cache directory')

args = parser.parse_args()
args.seed = int(args.seed)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(args.save), exist_ok=True)

# Delete existing results file if it exists
if os.path.exists(args.save):
    os.remove(args.save)

# Fixed test RNG for reproducible evaluation
test_rng = np.random.RandomState(42)

file_paths = ['claude.txt','o1.txt', 'flavors.txt','diverse.txt','artists.txt','movements.txt','subjects.txt', 'mediums.txt']

# Combine all files into one list
diverse_list = []
for file_path in file_paths:
    try:
        with open(file_path, 'r') as file:
            diverse_list.extend([line.strip() for line in file])
    except FileNotFoundError:
        print(f"Warning: Could not find file {file_path}")

# Remove duplicates while preserving order
diverse_list = list(dict.fromkeys(diverse_list))

if len(diverse_list) > 1000:
    print(f"Randomly capping diverse_list from {len(diverse_list)} to 1000 items")
    diverse_list = list(test_rng.choice(diverse_list, size=1000, replace=False))

# Split diverse_list into training (75%) and testing (25%) sets using fixed seed
n_train = int(0.75 * len(diverse_list))
indices = test_rng.permutation(len(diverse_list))
train_indices = indices[:n_train]
test_indices = indices[n_train:]

training_words_list = [diverse_list[i] for i in train_indices]
testing_words_list = [diverse_list[i] for i in test_indices]

horizon = 3
allowed_words_per_timestep = [training_words_list] * horizon
horizon = len(allowed_words_per_timestep)

# Initialize environment
if args.algorithm == "optim":
    words = cartesian(allowed_words_per_timestep)
    words_list = []
    for i in range(words.shape[0]):
        tokens = [t for t in words[i] if t != " "]
        pp = ", ".join(tokens)
        words_list.append(pp)
    env = LLMGrid(list_of_text_tokens=[words_list], MODELS_CACHE_DIR=args.cache_dir)
else:
    env = LLMGrid(list_of_text_tokens=allowed_words_per_timestep, MODELS_CACHE_DIR=args.cache_dir)

# Setup CLIP embedding and model
embedder, model = setup_clip_model(env)
theta_star = make_theta_star(env, embedder, model)

# Setup based on feedback type
m = 768
embedding = CustomEmbedding(m, lambda x: x, m)

if args.feedback_type == 'numerical':
    design = DesignA(env=env, lambd=args.lambda_reg, dim=1)
    estimator = KernelizedFeatures(embedding, m)
else:
    design = MultiPolicyOrigDesignD(env=env, lambd=args.lambda_reg, dim=1)
    likelihood = MultinomialLikelihood()
    regularizer = L2Regularizer(lam=args.lambda_reg)
    estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)

feedback = EmptyFeedback(env, design)
initial_policy = False

# Configure algorithm
if args.algorithm != 'random':
    if args.feedback_type == 'numerical':
        args.num_components = 750
    else:
        args.num_components = 75
else:
    initial_policy = True
    args.num_components = 1

num_policies = 1 if args.feedback_type == 'numerical' else 3

# Setup solver
convex_solver = FrankWolfe(
    env,
    objective=design,
    num_components=args.num_components,
    num_summarized_policies=num_policies,
    solver=DP,
    initial_policy=initial_policy,
    SummarizedPolicyType=DensityPolicy,
    accuracy=args.accuracy,
    step='line-search',
)

# Run exploration
if args.feedback_type == 'numerical':
    me = MdpExplore(
        env=env,
        objective=design,
        convex_solver=convex_solver,
        verbosity=3,
        feedback=feedback,
        general_policy='markovian'
    )
else:
    me = MdpExploreMultiPolicy(
        num_policies=num_policies,
        env=env,
        objective=design,
        convex_solver=convex_solver,
        verbosity=3,
        feedback=feedback,
        general_policy='markovian'
    )

val, opt_val, visits = me.run(episodes=args.episodes, return_visitations=True)

print('Finished exploration, estimating...')

prefix_range = range(1, horizon + 1) if args.dense_feedback else range(horizon, horizon + 1)
num_samples = args.episodes * (horizon if args.dense_feedback else 1)
trajectory_indices = torch.zeros((num_samples, horizon, num_policies), dtype=torch.long)

if args.feedback_type == 'numerical':
    x = []
    y = []
    for i in range(args.episodes):
        actions = visits[i][1]
        for prefix_len in prefix_range:
            truncated_actions = actions[:prefix_len]
            yy, xx = theta_star(truncated_actions, returnx=True)
            x.append(xx)
            y.append(yy)
    x = torch.vstack(x).detach()
    y = torch.vstack(y).detach()
    estimator.load_data((x, y))
    estimator.fit()
else:
    labels = torch.zeros((num_samples, num_policies))
    
    sample_idx = 0
    for t in range(args.episodes):
        policy_actions = [visits[p][t][1] for p in range(num_policies)]
        
        for prefix_len in prefix_range:
            # Store raw action indices, let theta_star handle token conversion
            truncated_actions = [actions[:prefix_len] + [0] * (horizon - prefix_len) for actions in policy_actions]
            
            vals = torch.tensor([theta_star(trunc) for trunc in truncated_actions])
            logits = torch.nn.functional.softmax(vals, dim=0)
            label = torch.multinomial(logits, 1)
            trajectory_indices[sample_idx,:,:] = torch.tensor(truncated_actions).T
            labels[sample_idx, label] = 1
            
            sample_idx += 1

    estimator.load_data((env.emissions, torch.zeros(len(env.emissions))))
    estimator.fit(trajectory_indices, labels, sum_dim=1)

print('Finished estimation, testing...')

N_test_prompts = 1000

# Generate test sequences
test_sequences = [generate_test_sequence(test_rng, testing_words_list, horizon) 
                 for _ in range(N_test_prompts)]

# Compute embeddings and predictions
xtest = []
ytest = []
for sequence in test_sequences:
    tokens = [t for t in sequence if t != " "]
    feat = embedder.embed_text('A plate with ' + ", ".join(tokens))
    yy = model(feat)
    xtest.append(feat)
    ytest.append(yy)

xtest = torch.vstack(xtest)
ytest = torch.vstack(ytest)
ypred = estimator.mean(xtest)

# Fixed sampling of test pairs
N_pairs_eval = 5000
pair_indices = np.array([(i, j) for i in range(N_test_prompts) for j in range(i+1, N_test_prompts)])
selected_pairs = pair_indices[test_rng.choice(len(pair_indices), N_pairs_eval, replace=False)]

correct_preferences = 0
for i, j in selected_pairs:
    gt_prefers_i = (ytest[i] >= ytest[j]).item()
    pred_prefers_i = (ypred[i] >= ypred[j]).item()
    if gt_prefers_i == pred_prefers_i:
        correct_preferences += 1

error = 1.0 - (correct_preferences / N_pairs_eval)

print('Finished estimation, saving accuracy to file...')

np.savetxt(args.save, np.array([[error]]))

print('Finished all, bye!')
