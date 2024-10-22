from doexpy.feedback.feedback_base import SimpleFeedback
import torch 
import numpy as np

class PoissonFeedback(SimpleFeedback):
    def __init__(self, env,
                 objective,
                 process, 
                 estimator = None,
                 problem = None,
                 opt = True) -> None:
        
        super().__init__(env, objective)
        
        self.env = env 
        self.opt = opt
        self.process = process
        self.objective = objective
        self.estimator = estimator
        self.problem = problem
        self.state_trajectory = []
        self.action_trajectory = []
        self.basic_sets = [env.sets[i] for i in np.arange(0,env.states_num,1)]

    def step_single(self, state: int, action: int):
        self.state_trajectory.append(state)
        self.action_trajectory.append(action)
        self.step_update()

    def step_episode(self):
        self.episode_update()
        # restart the trajectories
        self.state_trajectory = []
        self.action_trajectory = []
    
    def step_update(self):
        pass
        
    def episode_update(self):
        
        dt = 1. 

        # for all the states, sample the process                           
        for state in self.state_trajectory:
            S = self.env.sets[state]
            y = self.process.sample(S)
            m = self.estimator.get_m()
            n = torch.randn(size=(y.size()[0], m)).double() if y is not None else None
            datapoint = (S, n, dt)
            self.estimator.add_data_point(datapoint)
        
        self.estimator.fit_gp()
        if self.opt:
            self.objective.Sigma = torch.sqrt(torch.Tensor([float(self.estimator.ucb(s, dt = dt)) for s in self.basic_sets]).reshape(-1))
        else:
            self.objective.Sigma = torch.sqrt(torch.Tensor([float(self.problem.estimator.mean_set(s, dt = dt)) for s in self.basic_sets]).reshape(-1))

        