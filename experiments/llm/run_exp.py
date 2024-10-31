import torch
import argparse

from image_generator import StableDiffusionGenerator
from doexpy.env.llm import LLMGrid
from doexpy.functionals.doe_static_functionals import DesignA, DesignD
from doexpy.mdpexplore import MdpExplore
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

parser = argparse.ArgumentParser(description='Protein capacity experiment.')
parser.add_argument('--episodes', default=9, type=int, help='Name of the file')
parser.add_argument('--no_tokens', default=2, type=int, help='Name of the file')
parser.add_argument('--feedback', default='markovian', type=str, help='Name of the file')

args = parser.parse_args()

file_path = 'dummy.txt'

with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

T = args.episodes
env = LLMGrid(list_of_text_tokens = [words_list,words_list, words_list, words_list], verbose=True)

design = DesignA(
    env=env,
    lambd=1.,
    dim = 1, # this signifies the actions matter 
)

# define the convex solver
convex_solver = FrankWolfe(env,
                           objective=design,
                           num_components=100,
                           solver=DP,
                           #step=,
                           SummarizedPolicyType=DensityPolicy,
                           accuracy=1e-15)


# define the feedback class
feedback = BanditFeedback(env, design, )

initial_policy = False

me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver,
    verbosity=3,
    feedback=feedback,
    general_policy='markovian'
)

val, opt_val, visits = me.run(
    episodes=T,
    return_visitations = True
)
print (visits)

image_generator = StableDiffusionGenerator("CompVis/stable-diffusion-v1-4",
                                           device = 'cuda',
                                           image_height = 256,
                                           image_width = 256, num_inference_steps=200)
image_generator.resample_random()
images = []

for no, episode in enumerate(visits):
    states, actions = episode
    print (f"Trajectory {no}:",end = ' ')
    #print (actions)
    prompt = 'A plate with '
    for action in actions:
        prompt += env.unique_elements[action] +" "
    
    image = image_generator.sample(prompt)
    images.append(image)
    print (prompt)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 3)
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
plt.show()