import torch
import argparse
from PIL import Image

from image_generator import StableDiffusionGenerator

from doexpy.env.llm import LLMGrid
from doexpy.functionals.doe_static_functionals import DesignA, DesignD
from doexpy.mdpexplore import MdpExplore
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.feedback.bandit_feedback_state import BanditFeedbackState
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.polynomial_embedding import CustomEmbedding

parser = argparse.ArgumentParser(description='Protein capacity experiment.')
parser.add_argument('--episodes', default=9, type=int, help='Name of the file')
parser.add_argument('--no_tokens', default=2, type=int, help='Name of the file')
parser.add_argument('--feedback', default='markovian', type=str, help='Name of the file')

args = parser.parse_args()

file_path = 'dummy.txt'

with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

# number of episodes
T = args.episodes

# initializes the environment, the horizon is number of separates token lists 
env = LLMGrid(list_of_text_tokens = [words_list,words_list, words_list, words_list], verbose=True)

design = DesignA(
    env=env, 
    lambd=1., # regularization constant without any info
    dim = 1, # this signifies the actions matter for the desgin
)

# define the convex solver
convex_solver = FrankWolfe(env,
                           objective=design,
                           num_components=100, # number of FW steps
                           solver=DP, # type of RL solver
                           SummarizedPolicyType=DensityPolicy, # this type of policy summarizes the components from FW steps
                           accuracy=1e-15) # accuracy of the solver FW


# define the true function returning the reward
m = 768
# dummy theta_star
theta_star_vector = env.embed_text(["Cheese"]).double()
#theta_star = lambda actions:  env.embed_clip(actions.view(-1).int().tolist()).view(-1,m) @ theta_star_vector.T


image_generator = StableDiffusionGenerator("CompVis/stable-diffusion-v1-4",
                                            device = 'cuda',
                                            image_height = 256,
                                            image_width = 256,
                                            num_inference_steps=50)
image_generator.resample_random()

def theta_star(actions):
    prompt = 'A plate with '
    print (actions)
    for action in actions:
        prompt += env.unique_elements[int(action)]    

    # generate an image
    image = image_generator.sample(prompt)
    
    # for debugging purposes
    # from matplotlib import pyplot as plt
    # plt.imshow(image)
    # plt.show()

    # embed an image 
    image_pil = Image.fromarray(image)
    inputs = env._processor(images=image_pil, return_tensors="pt")
    feat = env._model.get_image_features(**inputs)

    # TODO: aesthetics model goes here
    val = feat.double() @ theta_star_vector.T
    return val 


# defines a custom Embedding for the estimator 
embedding = CustomEmbedding(1,env.embed_clip,m)
estimator = KernelizedFeatures(embedding, m)

# define the feedback class
feedback = BanditFeedbackState(env, design, estimator, theta_star, sigma=0.1, markovian=True)

# This stars without a policy, and first one is generated after first FW step
initial_policy = False

me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver, 
    verbosity=3, # verbosity level
    feedback=feedback,
    general_policy='markovian' # this means the feedback is called only after episode execution 
)

val, opt_val, visits = me.run(
    episodes=T,
    return_visitations = True # returns visitations as list of episodes with tuples of states and actions
)
