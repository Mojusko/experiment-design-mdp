
<img src="icon.png" alt="Icon" style="height: 4em; vertical-align: middle;"> <span style="font-size: 1em; vertical-align: middle;">DOExPy - An ML experiment design library</span>
-------------------------------------

This repo contains the efficient and elegant implementation of python library. It builds on numpy and cvxpy and stpy libraries. It provides efficient way to implement:

1. Bayesian Optimization, Maximum Identification
2. Active Learning and Exploration algorithms
3. Experiment Design, Reinforcement learning with advanced exploration directed loss-functions 
3. The library supports complex decision environments:\
i) State-less systems \
ii) Tabular MDPs\
iii) Linear Systems\
iv) Linear MDPs
    
The library was developed as part of multi-year process and was used in the following publications. 

1. [Active Exploration via Experiment Design in Markov Chains](https://proceedings.mlr.press/v206/mutny23a.html), Published at: *AISTATS 2023*
2. [Transition Constrained Bayesian Optimization via Markov Decision Processes
](https://arxiv.org/abs/2402.08406), Published at: *NeurIPS 2024*
3. [Modern Adaptive Experiment Design: Machine Learning Perspective](https://www.research-collection.ethz.ch/handle/20.500.11850/669541), *PhD Thesis of Mojmir Mutny*

**Example:**
Informative trajectories in the environment maximizing the diverse landscapes visited::

<img src="trajs.png" alt="Icon" style="height: 20em; vertical-align: middle;">

## 📦 Installation

**Dependencies**:

 [`stpy`](www.github.com/Mojusko/stpy) - Stochastic Process library for Python 
 
 `numpy` - Numerical LA

 `cvxpy` - Convex optimization 

**Via github**:

To ensure you have a compatible environment to run the code in, we recommend using the [`environment.yaml`](environment.yaml) file to create a conda environment using `conda` or `mamba`. If you are curious, the detailed dependencies are listed in [`pyproject.toml`](pyproject.toml).

```bash
pip install -e . 
```

## 🚀 Usage

A user needs to implement three things `environment`, `feedback model` and `functional` or also known as utility. 

 which is a class that tells given an action and state what are the probabilities or transitions to the next state. 

#### Environment 
An environment model provides on one had a transition function to simulate a transition i.e. $x' \sim p(x'|x,a)$, and also provides a way to calculate a probability of specific transition, namely, $p(x'|x,a)$. Each new environment needs to be newly implemented, but a user is encouraged to use LLMs for this task given the examples provided.

#### Feedback model 
Implements a unit that changes the utility after a feedback has been received. This was we can implement bandit or Bayesian optimization algorithm. For example, for maximum identification, the utility is

$$ U(d) \max_{z,z' \in \mathcal{Z}} ||\Phi(z)-\Phi(z')||_{(\sum_i d_i x_i x_i^\top)^{-1}}$$
where $\mathcal{Z}$ is a set of potential maximizers of the function which is defined as 

#### Utility
The utility function represent the task a user wants to achieve. We provide a list of classical utilities such as:
- Regret Maximization
- Maximum Identification (Best-arm identification)
- A-Experiment Design
- D-Experiment Design
- C-Experiment Design 
- V-Experiment Design 
- combinations thereof


#### The solution
We propose to solve the utility as a function of the policy or its state-visitation in this case. This is not always possible, but is at the core of our approach. All the above problems satisfy this (see [[1]](https://proceedings.mlr.press/v206/mutny23a.html)). This can be either solved using interior-point solver if state-action polytope (constraints of the environment) can be summarized as a polytope. If this is not the case, we propose to use the *Frank-Wolfe*, where individual linear maximization oracles correspond to classical reinforcement learning that can be solved using myriad ways: 
- DP (dynamic programming)
- LP (linear programming)
- Policy iteration 
- Value iteration (suitably implemented)
- For continuous state-spaces: Model predictive control, or different reparametrization techniques. 

### Easy Example 
```python

# Initialize the environment, in this case sochastic grid
env = StochasticDummyGridWorld(prob = args.probability)

# Choose the goal (utility) often referred to as functional in the code
if adaptive:
    design = AdaptiveDesignD(env, lambd=1e-3)  # Adaptive design, that performs iterative planning and see executed trajectories
else:
    design = DesignD(env, lambd=1e-3)  # Standard design, that is static and executed at once

# Define the convex solver using the Frank-Wolfe algorithm
convex_solver = FrankWolfe(
    env, 
    objective=design, # defign functiona
    num_components=args.num_components,  # Number of linearizations for the Frank-wolfe algorithm
    solver=DP,  # Solver for linearized problems (Dynamic programming here)
    SummarizedPolicyType=DensityPolicy,  # Type of policy summarization and/or execution
    accuracy=args.accuracy  # Dual certificate of Frank-Wolfe
    )

# Define the feedback class - In this case there is no feedback as D-design is without unknowns
feedback = EmptyFeedback(env, design)

# Create an instance of MdpExplore with the specified parameters
me = MdpExplore(
    env=env,
    objective=design,
    convex_solver=convex_solver, # Frank-Wolfe or Interior Point
    verbosity=args.verbosity,
    feedback=feedback, 
    general_policy='markovian'  # Type of general policy
)
# A non-markovian policy performs updates to itself after each transition and replans. Markovian policy calculated a fixed markovian policy per episode (feedback iteration)

# Run the MdpExplore instance
val, opt_val = me.run(
    episodes=args.episodes,  # Number of sequential interactions (experimental budget)
    save_trajectory=args.savetrajectory  # Whether to save the trajectory
)

# The function returns the values and trajectories that were executed
```




## 📜 License

This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📃 Citing this work

Please cite one of our papers if you use this code in your work. For example, see below: 

```bibtex
@inproceedings{Mutny2023,
	abstract = {A key challenge in science and engineering is to design experiments to learn about some unknown quantity of interest. Classical experimental design optimally allocates the experimental budget to maximize a notion of utility (e.g., reduction in uncertainty about the unknown quantity). We consider a rich setting, where the experiments are associated with states in a {\em Markov chain}, and we can only choose them by selecting a {\em policy} controlling the state transitions. This problem captures important applications, from exploration in reinforcement learning to spatial monitoring tasks. We propose an algorithm -- \textsc{markov-design} -- that efficiently selects policies whose measurement allocation \emph{provably converges to the optimal one}. The algorithm is sequential in nature, adapting its choice of policies (experiments) informed by past measurements. In addition to our theoretical analysis, we showcase our framework on applications in ecological surveillance and pharmacology.},
	author = {Mutn{\'y}, Mojm{\'\i}r and Janik, Tadeusz and Krause, Andreas},
	booktitle = {{Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)}},
	howpublished = {AISTATS 2023},
	keywords = {Machine Learning (cs.LG); Methodology (stat.ME); Machine Learning (stat.ML); FOS: Computer and information sciences},
	title = {{Active Exploration via Experiment Design in Markov Chains}},
	url = {https://arxiv.org/abs/2206.14332},
	year = {2023}
}
```

## 👥 Authors & Contributors

- [Mojmir Mutny](https://mojmirmutny.github.io/)
- [Tadeusz Janik](https://www.linkedin.com/in/tadeusz-janik/?originalSubdomain=uk)
- [Jose Pablo Folch](https://jpfolch.github.io/)
- [Weronica Oromaniec](https://www.linkedin.com/in/weronika-ormaniec/?locale=en_US)

## 📧 Contact

For questions, please contact
- mojmir.mutny@inf.ethz.ch

## 🤝 Contributing

We welcome contributions to this repository. To set up the development environment, please follow the instructions below:



### Conventions 
    - the stored discrete visitations are Hx|S|x|A|

