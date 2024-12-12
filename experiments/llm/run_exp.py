import torch
import argparse
from stpy.helpers.helper import cartesian
from PIL import Image
import time
from image_generator import StableDiffusionGenerator

import os
from doexpy.env.llm import LLMGrid

from doexpy.functionals.doe_static_functionals import DesignA, DesignD
from doexpy.mdpexplore import MdpExplore
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.feedback.bandit_feedback_state import BanditFeedbackState
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

# IF DOES NOT WORK UNCOMMEND THE FOLLOWING LINE
from stpy.regression.kernelized_features import KernelizedFeatures
#from stpy.continuous_process.kernelized_features import KernelizedFeatures


from stpy.embeddings.polynomial_embedding import CustomEmbedding
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='LLM experiment.')
parser.add_argument('--episodes', default=10, type=int, help='Episodes')
parser.add_argument('--no_tokens', default=2, type=int, help='Horizon')
parser.add_argument('--feedback', default='markovian', type=str, help='what type of feedback (end) mid episode')
parser.add_argument('--algorithm', default='greedy', type=str, help='type of algorithm')
parser.add_argument('--num_components', default=100, type=int, help='Number of components in FW')
parser.add_argument('--save', default="results/experiment.csv", type=str, help='name of the file')
parser.add_argument('--seed', default=12, type=str, help='Use this to set the seed for the random number generator')
parser.add_argument('--accuracy', default=None, type=float, help='Termination criterion for optimality gap')
parser.add_argument('--opt', default=None, type=str, help='whether to return opt')
parser.add_argument('--store_folder', default='/cluster/project/krause/mmutny', type=str, help='whether to return opt')

args = parser.parse_args()

args.seed = int(args.seed)


file_path = 'mediums_small.txt'
with open(file_path, 'r') as file:
    words_list_1 = [line.strip() for line in file]

file_path = 'movements_small.txt'
with open(file_path, 'r') as file:
    words_list_2 = [line.strip() for line in file]

# number of episodes
T = args.episodes
os.environ['TORCH_HOME'] =args.store_folder

# create a cartesian version of the word_list
words = cartesian([words_list_1,words_list_2])
words_list = []

for i in range(words.shape[0]):
    if words[i][0] != " ":
        pp = words[i][0]+", "+words[i][1]
    else:
        pp = words[i][1]
    words_list.append(pp)

if args.algorithm == "optim":
    env = LLMGrid(list_of_text_tokens = [words_list], verbose=True, MODELS_CACHE_DIR = args.store_folder)

else:
    # initializes the environment, the horizon is number of separates token lists
    env = LLMGrid(list_of_text_tokens = [words_list_1,words_list_2], verbose=True, MODELS_CACHE_DIR = args.store_folder)

design = DesignA(
    env=env, 
    lambd=1., # regularization constant without any info
    dim = 1, # this signifies the actions matter for the design
)

# define the true function returning the reward
m = 768
# dummy theta_star
theta_star_vector = env.embed_text(["Cheese"]).double()
#theta_star = lambda actions:  env.embed_clip(actions.view(-1).int().tolist()).view(-1,m) @ theta_star_vector.T


# image_generator = StableDiffusionGenerator("CompVis/stable-diffusion-v1-4",
#                                             device = 'cuda',
#                                             image_height = 256,
#                                             image_width = 256,
#                                             num_inference_steps=50)
# image_generator.resample_random()


model = nn.Linear(768, 1).double()
state = torch.load("vit_14_weights.pth")
model.load_state_dict(state)
model.eval()

# this is shortcut for working with
def embed_clip(prompt):
    text_input = env._tokenizer(
        prompt,
        padding="max_length",
        max_length=env._tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    feat = env._model.get_text_features(**text_input)  # projected CLIP embeddings
    feat = feat.detach().double().view(1,-1)
    return feat

def theta_star(actions, returnx = False):
    prompt = 'A plate with '
    print (actions)
    added = None
    for action in actions:
        if added is None:
            added = env.unique_elements[int(action)]
        else:
            if added != " ":
                added += ", " + env.unique_elements[int(action)]
            else:
                added = env.unique_elements[int(action)]
    prompt = prompt + added

    # TODO: Uncomment when working with images

    # # generate an image
    # image = image_generator.sample(prompt)
    #
    # # for debugging purposes
    # # from matplotlib import pyplot as plt
    # # plt.imshow(image)
    # # plt.show()
    #
    # # embed an image
    # image_pil = Image.fromarray(image)
    # inputs = env._processor(images=image_pil, return_tensors="pt")
    # feat = env._model.get_image_features(**inputs)

    text_input = env._tokenizer(
        prompt,
        padding="max_length",
        max_length=env._tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    feat = env._model.get_text_features(**text_input)  # projected CLIP embeddings
    feat = feat.detach().double()
    print ("Evaluating prompt", prompt)
    val = model.forward(feat)
    if returnx:
        return val, feat
    else:
        return val


# defines a custom Embedding for the estimator
embedding = CustomEmbedding(m,lambda x: x,m)
estimator = KernelizedFeatures(embedding, m)

# define the feedback class
# TODO: This feedback is when using non-Markovina or directed design
#feedback = BanditFeedbackState(env, design, estimator, theta_star, sigma=0.1, markovian=True)

# This is for designs not requiring any feedback
feedback = EmptyFeedback(env, design)

# This stars without a policy, and first one is generated after first FW step
initial_policy = False

if args.algorithm == 'greedy':
    pass
    args.num_components = 100
elif args.algorithm == "optim":
    pass
    args.num_components = 1000
elif args.algorithm == "random":
    initial_policy = True
    args.num_components = 1
else:
    raise NotImplementedError("This algorithm is not implemented yet.")

# define the convex solver
convex_solver = FrankWolfe(env,
                           objective=design,
                           num_components=args.num_components, # number of FW steps
                           solver=DP, # type of RL solver
                           step = "line-search",
                           initial_policy=initial_policy, # initial policy
                           SummarizedPolicyType=DensityPolicy, # this type of policy summarizes the components from FW steps
                           accuracy=args.accuracy) # accuracy of the solver FW

me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver,
    verbosity=3,  # verbosity level
    feedback=feedback,
    general_policy='markovian'  # this means the feedback is called only after episode execution
)

val, opt_val, visits = me.run(
    episodes=T,
    return_visitations = True # returns visitations as list of episodes with tuples of states and actions
)

# These are the visited trajectories of the model
print (visits)

### Testing part of the code #######
####################################

# Lets generate a feedback for the estimator from the asthetics model
x = []
y = []
actions = []
for i in range(T):
    action = visits[i][1]
    actions.append(action)
    yy,xx = theta_star(action, returnx = True)
    x.append(xx)
    y.append(yy)

x = torch.vstack(x).detach()
y = torch.vstack(y).detach()

# load data
estimator.load_data((x,y))

# fit the model
estimator.fit()

# sample random list of words
N_random = 200
selected_words = np.random.choice(words_list, N_random)

xtest = []
ytest = []
for i in range(len(selected_words)):
    t1 = time.time()
    fea = embed_clip(selected_words[i])
    #print ("i", time.time()-t1)
    yy = model(fea)
    #print("i", time.time() - t1)
    #print ("-----")
    xtest.append(fea)
    ytest.append(yy)

xtest = torch.vstack(xtest)
ytest = torch.vstack(ytest)

ypred = estimator.mean(xtest)
error = torch.mean((ypred - ytest)**2)

vals = np.array(val)
np.savetxt(args.save, error.detach().view(1,1).numpy())
