import torch
import os
import argparse
from stpy.helpers.helper import cartesian
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

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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

set_all_seeds(args.seed)

# Load word lists
#file_path = 'mediums_small.txt'
#file_path = 'mediums.txt'
file_path = 'mediums_large.txt'
with open(file_path, 'r') as file:
    words_list_1 = [line.strip() for line in file]

#file_path = 'movements_small.txt'
#file_path = 'movements.txt'
file_path = 'movements_large.txt'
with open(file_path, 'r') as file:
    words_list_2 = [line.strip() for line in file]

file_path = 'subjects.txt'
with open(file_path, 'r') as file:
    words_list_3 = [line.strip() for line in file]

#words_list_1 = words_list_1[:10]
#words_list_2 = words_list_2[:10]
#words_list_3 = words_list_3[:10]

all_words_lists = [words_list_1,words_list_2, words_list_3]
horizon = len(all_words_lists)


# Initialize environment
if args.algorithm == "optim":
    # Only calculate cartesian product for optim algorithm
    words = cartesian(all_words_lists)
    words_list = []
    for i in range(words.shape[0]):
        tokens = [t for t in words[i] if t != " "]
        pp = ", ".join(tokens)
        words_list.append(pp)
    env = LLMGrid(list_of_text_tokens=[words_list], MODELS_CACHE_DIR=args.cache_dir)
else:
    env = LLMGrid(list_of_text_tokens=all_words_lists, MODELS_CACHE_DIR=args.cache_dir)

# Load aesthetics model
model = nn.Linear(768, 1).double()
state = torch.load("vit_14_weights.pth")
model.load_state_dict(state)
model.eval()

def embed_clip(prompt):
    text_input = env._tokenizer(
        prompt,
        padding="max_length",
        max_length=env._tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    feat = env._model.get_text_features(**text_input)
    feat = feat.detach().double().view(1,-1)
    return feat

def theta_star(actions, returnx=False, verbose=False):
    # No actions case should not occur with proper truncation
    assert len(actions) > 0  
    
    prompt = 'A plate with '
    tokens = [env.unique_elements[int(action)] for action in actions]
    valid_tokens = [t for t in tokens if t != " "]
    
    if valid_tokens:
        prompt += ", ".join(valid_tokens)

    if verbose:
        print(prompt)

    text_input = env._tokenizer(
        prompt,
        padding="max_length",
        max_length=env._tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    feat = env._model.get_text_features(**text_input)
    feat = feat.detach().double()
    val = model.forward(feat)
    if returnx:
        return val, feat
    else:
        return val

# Setup based on feedback type
m = 768
embedding = CustomEmbedding(m, lambda x: x, m)

if args.feedback_type == 'numerical':
    design = DesignA(env=env, lambd=args.lambda_reg, dim=1)
    estimator = KernelizedFeatures(embedding, m)
else:
    #design = MultiPolicyAggDesignA(env=env, lambd=1., dim=1)
    #design = MultiPolicyAggDesignD(env=env, lambd=1., dim=1)
    design = MultiPolicyOrigDesignD(env=env, lambd=args.lambda_reg, dim=1)
    likelihood = MultinomialLikelihood()
    regularizer = L2Regularizer(lam=args.lambda_reg)
    estimator = RegularizedMultinomialEstimator(embedding, likelihood, regularizer)

feedback = EmptyFeedback(env, design)
initial_policy = False

# Configure algorithm
if args.algorithm != 'random':
    if args.feedback_type == 'numerical':
        args.num_components = 250
    else:
        # we have multiple rounds, don't need many iterations
        args.num_components = 50

else:
    initial_policy = True
    args.num_components = 1

#if args.algorithm == "optim":
#    args.num_components = 200
#else:
#    raise NotImplementedError("This algorithm is not implemented yet.")

if args.feedback_type == 'numerical':
    num_policies=1
else:
    num_policies=2

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

# Fit estimator based on feedback type
#if args.feedback_type == 'numerical':
#    x = []
#    y = []
#    for i in range(args.episodes):
#        action = visits[i][1]
#        yy, xx = theta_star(action, returnx=True)
#        x.append(xx)
#        y.append(yy)
#    
#    x = torch.vstack(x).detach()
#    y = torch.vstack(y).detach()
#    estimator.load_data((x, y))
#    estimator.fit()
#else:
#    estimator.load_data((env.emissions, torch.zeros(len(env.emissions))))
#    labels = torch.zeros((args.episodes, 2))
#    trajectory_indices = torch.zeros((args.episodes, horizon, 2), dtype=torch.long)
#    
#    for t in range(args.episodes):
#        trajectory_indices[t,:,0] = torch.tensor(visits[0][t][1])
#        trajectory_indices[t,:,1] = torch.tensor(visits[1][t][1])
#        
#        val1 = theta_star(visits[0][t][1])
#        val2 = theta_star(visits[1][t][1])
#        logits = torch.tensor([val1, val2])
#        label = torch.multinomial(torch.nn.functional.softmax(logits, dim=0), 1)
#        labels[t, label] = 1
#    

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


# Evaluation
N_random = 100
N_pairs_pme = 500  # Number of pairs to evaluate preference alignment

# First randomly sample from individual word lists, with possibility of " "
selected_combinations = []
for _ in range(N_random):
    combination = []
    for word_list in all_words_lists:
        # For each position, either pick a word from word_list or " "
        if np.random.random() < 0.1:  # 20% chance of picking " "
            word = " "
        else:
            word = np.random.choice(word_list)
        combination.append(word)
    selected_combinations.append(combination)

# Now create the combined strings
xtest = []
ytest = []
for combo in selected_combinations:
    tokens = [t for t in combo if t != " "]  # Using existing filtering
    prompt = 'A plate with ' + ", ".join(tokens)
    fea = embed_clip(prompt)
    yy = model(fea)
    xtest.append(fea)
    ytest.append(yy)

xtest = torch.vstack(xtest)
ytest = torch.vstack(ytest)

ypred = estimator.mean(xtest)


# Sample random pairs and compute preference alignment
correct_preferences = 0
# Sample N_pairs_pme unique pairs
pair_indices = np.array([(i, j) for i in range(N_random) for j in range(i+1, N_random)])
selected_pairs = pair_indices[np.random.choice(len(pair_indices), N_pairs_pme, replace=False)]

for i, j in selected_pairs:
    # Get ground truth preference
    gt_prefers_i = (ytest[i] >= ytest[j]).item()
    
    # Get predicted preference
    pred_prefers_i = (ypred[i] >= ypred[j]).item()
    
    # Check if preferences align
    if gt_prefers_i == pred_prefers_i:
        correct_preferences += 1

# Calculate preference alignment error (percentage of misaligned preferences)
error = 1.0 - (correct_preferences / N_pairs_pme)

print('Finished estimation, saving accuracy to file...')

np.savetxt(args.save, np.array([[error]]))

print('Finished all, bye!')

