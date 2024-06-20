
# ![workflow](icon.png) DOExPy - An ML experiment design library

This repo contains the efficient and elegant implementation of python library. It builds on numpy and cvxpy and stpy libraries. It provides efficient way to implement:

1. Bayesian Optimization, Maximum Identification
2. Active Learning and Exploration algorithms
3. Reinforcement learning with advanced exploration directed loss-functions 
3. The library supports complex decision environments
    a. State-less systems
    b. Tabular MDPs
    c. Linear Systems
    d. Linear MDPs
    
The library was developed as part of multi-year process and was used in the following publications. 

1. [Active Exploration via Experiment Design in Markov Chains](https://proceedings.mlr.press/v206/mutny23a.html), Published at: *AISTATS 2023*
2. [Transition Constrained Bayesian Optimization via Markov Decision Processes
](https://arxiv.org/abs/2402.08406), Published at: *NeurIPS 2024*
3. [Modern Adaptive Experiment Design: Machine Learning Perspective](https://www.research-collection.ethz.ch/handle/20.500.11850/669541), *PhD Thesis of Mojmir Mutny*


![workflow](heatmap.png)
![workflow](trajs.png)

## 📦 Installation

**Dependencies**:

 stpy

**Via github**:

To ensure you have a compatible environment to run the code in, we recommend using the [`environment.yaml`](environment.yaml) file to create a conda environment using `conda` or `mamba`. If you are curious, the detailed dependencies are listed in [`pyproject.toml`](pyproject.toml).

```bash
pip install git+https://github.com/Hollfelder-Lab/lrDMS-IRED.git
```

## 🚀 Usage

#### Environemnt 
A user has to implement an environment 

#### Feedback model 


#### Convex Solver 
    - Frank-wolfe
    - Interior-Point method 

#### Solver
Classical solver solving a linearized variant of the loss function. 

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

