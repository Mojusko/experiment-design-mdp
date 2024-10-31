import numpy as np
from typing import Union, Callable
import torch
from scipy.optimize import minimize
from scipy.linalg import norm
from doexpy.feedback.feedback_base import SimpleFeedback
from doexpy.env.bandits import MovementConstrainedBayesianOptimization
from doexpy.functionals.reward_functional import RewardFunctional
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.continuous_processes.kernelized_features import KernelizedFeatures



class ContinuousBanditFeedback(SimpleFeedback):
    def __init__(self, 
                env:MovementConstrainedBayesianOptimization, 
                objective:RewardFunctional, 
                estimator:Union[GaussianProcess, KernelizedFeatures], 
                theta_star:Union[np.array, Callable], 
                sigma:float, 
                video:bool = False, 
                markovian:bool = False,
                initial_point: np.array = None,
                maximization_set_method:str = 'thompson_sampling',
                keep_track_best_guess: bool = False) -> None:
        
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedding = self.objective.embedding
        self.video = video
        self.markovian = markovian
        self.maximization_set_method = maximization_set_method
        if self.maximization_set_method == 'ucb_grid':
            self.create_maximizers_grid()
        elif self.maximization_set_method == 'none':
            pass
        # define the initial point
        if initial_point is None:
            self.initial_point = np.ones((1, self.env.states_dim)) * -0.5
        else:
            self.initial_point = initial_point
        self.initialize(initial_point)
        # keep track of best guess
        self.keep_track_best_guess = keep_track_best_guess
        self.best_arm = []
    
    def initialize(self, initial_point):
        eps = np.random.normal(0, self.sigma**2)
        if callable(self.theta_star):
            fun_value = self.theta_star(initial_point)
        else:
            z = self.embedding.embed(torch.tensor(initial_point)).numpy()
            eps = np.random.normal(0, self.sigma)
            fun_value = z @ self.theta_star + eps
        
        # add the data point to the queue
        data_point = initial_point, fun_value

        # add the data point to the estimator
        self.estimator.add_data_point(torch.tensor(data_point[0]),
                                        torch.tensor([[data_point[1]]]))
        # fit the estimator
        self.estimator.fit()

    def thompson_sample_potential_maximizers(self, num_maximizers = None):
        if type(self.estimator) == KernelizedFeatures:
            if num_maximizers is None:
                num_maximizers = self.objective.num_of_maximizers

            # sample thetas from the posterior and maximize
            maximizers, _ = self.estimator.sample_and_optimize(size = num_maximizers)
            
            return maximizers.numpy()

        else:
            raise NotImplementedError('Thompson sampling for GPs not implemented yet')
    
    def ucb_potential_maximizers(self, num_maximizers = None, betas = None):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        if betas is None:
            betas = np.linspace(0, 2.5, num_maximizers)

        for max_idx in range(num_maximizers):
            # optimize the UCB acquisition function for each beta
            beta = betas[max_idx]

            def f(x):
                return -self.UCB(x, beta = beta)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers
    
    def ei_potential_maximizers(self, num_maximizers = 1):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        for max_idx in range(num_maximizers):
            # optimize the EI acquisition function for each beta

            def f(x):
                return -self.EI(x)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers

    def pi_potential_maximizers(self, num_maximizers = 1):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        for max_idx in range(num_maximizers):
            # optimize the PI acquisition function for each beta
            def f(x):
                return -self.PI(x)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers
    
    def af_mix_potential_maximizers(self, num_maximizers = 6):
        # obtain the maximizers from the different acquisition functions
        ucb_maximizers = self.ucb_potential_maximizers(num_maximizers = 3, betas = [0, 1, 2])
        ei_maximizers = self.ei_potential_maximizers(num_maximizers = 1)
        pi_maximizers = self.pi_potential_maximizers(num_maximizers = 1)
        ts_maximizers = self.thompson_sample_potential_maximizers(num_maximizers = num_maximizers - 5)
        # concatenate them
        potential_maximizers = np.vstack((ucb_maximizers, ei_maximizers, pi_maximizers, ts_maximizers))
        
        return potential_maximizers
    
    def create_maximizers_grid(self):
        if self.env.states_dim == 1:
            self.maximizers_grid = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)
        elif self.env.states_dim == 2:
            self.maximizers_grid = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))).T.reshape(-1, 2)
        else:
            raise NotImplementedError('Maximizers grid for higher dimensions not implemented yet')

    def ucb_grid_potential_maximizers(self):
        
        lcbs = self.estimator.lcb(torch.tensor(self.maximizers_grid)).numpy().squeeze()
        highest_lcb = np.max(lcbs)

        ucbs = self.estimator.ucb(torch.tensor(self.maximizers_grid)).numpy().squeeze()

        return self.maximizers_grid[ucbs >= highest_lcb]
    
    def step_update(self):
        if self.markovian:
            pass
        else:
            action = self.action_trajectory[-1]
            # see the next state
            next_state = self.env.next(self.env.state, action)
            print('next design: ', next_state)
            # obtain the corresponding observation
            eps = np.random.normal(0, self.sigma**2)
            if callable(self.theta_star):
                fun_value = self.theta_star(next_state)
            else:
                z = self.embedding.embed(torch.tensor(next_state)).numpy()
                eps = np.random.normal(0, self.sigma)
                fun_value = z @ self.theta_star + eps

            self.estimator.add_data_point(torch.tensor(next_state),
                                        torch.tensor(fun_value).reshape((-1, 1)))
            self.estimator.fit()

            # keep track of best guess
            if self.keep_track_best_guess:
                best_guess = self.optimum_location_guess()
                self.best_arm.append(best_guess)

            if self.video:
                # save the relevant stuff for plotting
                memory_states = np.load('states.npy')
                memory_maximizers = np.load('maximizers.npy')
                memory_mean = np.load('mean.npy')
                memory_ucb = np.load('ucb.npy')
                memory_lcb = np.load('lcb.npy')

                # need to save the state
                np.save('states.npy', np.vstack((memory_states, np.array(next_state))))

                # need to save the set of maximizers
                # create copy of the set of maximizers
                maximizers = self.objective.set_of_maximizers.copy()

                # extend with empty arrays to match the number of maximizers
                # maximizers = np.vstack((maximizers, np.zeros((self.objective.num_of_maximizers - maximizers.shape[0], maximizers.shape[1]))))

                np.save('maximizers.npy', np.vstack((memory_maximizers, np.expand_dims(maximizers, 0))))

                if self.env.states_dim == 1:
                    # need to save the state of the GP
                    discrete_state_space = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)

                    lcb = self.estimator.lcb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    ucb = self.estimator.ucb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    mean = self.estimator.mean(torch.tensor(discrete_state_space)).numpy().squeeze()

                    np.save('mean.npy', np.vstack((memory_mean, mean)))
                    np.save('ucb.npy', np.vstack((memory_ucb, ucb)))
                    np.save('lcb.npy', np.vstack((memory_lcb, lcb)))

            if self.maximization_set_method == 'ucb_grid':
                self.objective.set_of_maximizers = self.ucb_grid_potential_maximizers()
            elif self.maximization_set_method == 'ucb':
                self.objective.set_of_maximizers = self.ucb_potential_maximizers()
            elif self.maximization_set_method == 'thompson_sampling':
                self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
            elif self.maximization_set_method == 'af_mix':
                self.objective.set_of_maximizers = self.af_mix_potential_maximizers()
            elif self.maximization_set_method == 'none':
                pass
            else:
                raise NotImplementedError('Maximization set method not implemented yet')

    def episode_update(self):
        if self.markovian:
            final_state = self.env.next(self.env.state, self.action_trajectory[-1])
            state_list = self.state_trajectory[1:] + [final_state]
            print('actions taken: ', state_list)

            for state in state_list:
                eps = np.random.normal(0, self.sigma**2)
                if callable(self.theta_star):
                    fun_value = self.theta_star(state) + eps
                else:
                    z = self.embedding.embed(torch.tensor(state)).numpy()
                    fun_value = z @ self.theta_star + eps
                
                self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                        torch.tensor([[fun_value]]))
            
            self.estimator.fit()
            if self.maximization_set_method == 'ucb':
                self.objective.set_of_maximizers = self.ucb_potential_maximizers()
            elif self.maximization_set_method == 'ucb_grid':
                self.objective.set_of_maximizers = self.ucb_grid_potential_maximizers()    
            elif self.maximization_set_method == 'thompson_sampling':
                self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
            elif self.maximization_set_method == 'af_mix':
                self.objective.set_of_maximizers = self.af_mix_potential_maximizers()
            elif self.maximization_set_method == 'none':
                pass

            # keep track of best guess
            if self.keep_track_best_guess:
                best_guess = self.optimum_location_guess()
                self.best_arm.append(best_guess)

            if self.video:
                # save the relevant stuff for plotting

                # need to save the state

                # need to save the set of maximizers

                # need to save the state of the GP

                pass
        
        else:
            pass
    
    def optimum_location_guess(self):
        '''
        Estimate the location of the optimum by maximizing the posterior mean
        '''
        # if the estimator is not fitted, return the initial point
        if not self.estimator.fitted:
            return self.initial_point


        # define the function to optimize
        def f(x):
            self.estimator.mean(torch.tensor(x.reshape(1, -1)))
            return -self.estimator.mean(torch.tensor(x.reshape(1, -1))).numpy().squeeze()
        
        # define the bounds
        bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)]

        # initialize the optimizer at the best observed point so far
        x0_idx = np.argmax(self.estimator.y)
        x0 = self.estimator.x[x0_idx, :].numpy()

        # run the optimization
        res = minimize(f, x0 = x0, bounds = bounds, method = 'L-BFGS-B', options = {'maxiter': 1000})

        return res.x.reshape(1, -1)
    
    def EI(self, x):
        '''
        Expected improvement acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # obtain the best observed value
        best_obs = np.max(self.estimator.y.numpy())
        # calculate the improvement
        improvement = mean.numpy().squeeze() - best_obs
        # calculate the expected improvement
        expected_improvement = improvement * norm.cdf(improvement / std.numpy().squeeze()) + std.numpy().squeeze() * norm.pdf(improvement / std.numpy().squeeze())
        # return the negative expected improvement
        return expected_improvement

    def PI(self, x):
        '''
        Probability of improvement acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # obtain the best observed value
        best_obs = np.max(self.estimator.y.numpy())
        # calculate the improvement
        improvement = mean.numpy().squeeze() - best_obs
        # calculate the probability of improvement
        probability_improvement = norm.cdf(improvement / std.numpy().squeeze())
        # return the negative probability of improvement
        return probability_improvement
    

    
    def UCB(self, x, beta = 2):
        '''
        Upper confidence bound acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # calculate the upper confidence bound
        ucb = mean.numpy().squeeze() + beta * std.numpy().squeeze()
        # return the negative upper confidence bound
        return ucb
        




class ContinuousBanditFeedbackAsynchronous(SimpleFeedback):
    def __init__(self, 
                env:MovementConstrainedBayesianOptimization, 
                objective:RewardFunctional, 
                estimator:Union[GaussianProcess, KernelizedFeatures], 
                theta_star:Union[np.array, Callable], 
                sigma:float, 
                video:bool = False, 
                markovian:bool = False,
                maximization_set_method:str = 'thompson_sampling',
                asynchronous_delay:int = 1,
                initial_point: np.array = None,
                keep_track_best_guess: bool = False) -> None:
        
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedding = self.objective.embedding
        self.video = video
        self.markovian = markovian
        self.maximization_set_method = maximization_set_method
        if self.maximization_set_method == 'ucb_grid':
            self.create_maximizers_grid()
        self.asynchronous_delay = asynchronous_delay
        # start a queue for the delay in the feedback
        self.queue = []
        # define the initial point
        if initial_point is None:
            self.initial_point = np.ones((1, self.env.states_dim)) * -0.5
        else:
            self.initial_point = initial_point
        self.initialize(initial_point)
        # keep track of best guess
        self.keep_track_best_guess = keep_track_best_guess
        self.best_arm = []
    
    def initialize(self, initial_point):
        eps = np.random.normal(0, self.sigma**2)
        if callable(self.theta_star):
            fun_value = self.theta_star(initial_point)
        else:
            z = self.embedding.embed(torch.tensor(initial_point)).numpy()
            eps = np.random.normal(0, self.sigma)
            fun_value = z @ self.theta_star + eps
        
        # add the data point to the queue
        self.queue.append((initial_point, fun_value))

        if len(self.queue) == self.asynchronous_delay:
            # add the data point to the estimator
            data_point = self.queue[0]
            self.estimator.add_data_point(torch.tensor(data_point[0]),
                                            torch.tensor([[data_point[1]]]))
            # fit the estimator
            self.estimator.fit()
            # update the maximizers
            if self.maximization_set_method == 'ucb':
                self.objective.set_of_maximizers = self.ucb_potential_maximizers()
            elif self.maximization_set_method == 'thompson_sampling':
                self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
            else:
                raise NotImplementedError('Maximization set method not implemented yet')
            # shorten the queue
            self.queue = self.queue[1:]

    def thompson_sample_potential_maximizers(self):
        if type(self.estimator) == KernelizedFeatures:

            # sample thetas from the posterior and maximize
            maximizers, _ = self.estimator.sample_and_optimize(size = self.objective.num_of_maximizers)
            
            return maximizers.numpy().reshape(self.objective.num_of_maximizers, self.env.states_dim)

        else:
            raise NotImplementedError('Thompson sampling for GPs not implemented yet')
    
    def ucb_potential_maximizers(self, num_maximizers = None, betas = None):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        if betas is None:
            betas = np.linspace(0, 2.5, num_maximizers)

        for max_idx in range(num_maximizers):
            # optimize the UCB acquisition function for each beta
            beta = betas[max_idx]

            def f(x):
                return -self.UCB(x, beta = beta)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers
    
    def ei_potential_maximizers(self, num_maximizers = 1):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        for max_idx in range(num_maximizers):
            # optimize the EI acquisition function for each beta

            def f(x):
                return -self.EI(x)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers

    def pi_potential_maximizers(self, num_maximizers = 1):
        if num_maximizers is None:
            num_maximizers = self.objective.num_of_maximizers
        
        potential_maximizers = np.zeros((num_maximizers, self.env.states_dim))

        for max_idx in range(num_maximizers):
            # optimize the PI acquisition function for each beta
            def f(x):
                return -self.PI(x)

            x0_idx = np.argmax(self.estimator.y)
            x0 = self.estimator.x[x0_idx, :].numpy().reshape(-1)

            # run the optimization with a few random restarts
            best_f = 1e10
            for _ in range(5):
                res = minimize(f, x0 = x0, bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)], method = 'L-BFGS-B', options = {'maxiter': 1000})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                # set x0 after so that first iteration is not random
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)

            potential_maximizers[max_idx, :] = best_x
        
        return potential_maximizers
    
    def af_mix_potential_maximizers(self, num_maximizers = 6):
        # obtain the maximizers from the different acquisition functions
        ucb_maximizers = self.ucb_potential_maximizers(num_maximizers = 3, betas = [0, 1, 2])
        ei_maximizers = self.ei_potential_maximizers(num_maximizers = 1)
        pi_maximizers = self.pi_potential_maximizers(num_maximizers = 1)
        ts_maximizers = self.thompson_sample_potential_maximizers(num_maximizers = num_maximizers - 5)
        # concatenate them
        potential_maximizers = np.vstack((ucb_maximizers, ei_maximizers, pi_maximizers, ts_maximizers))
        
        return potential_maximizers
        
    def create_maximizers_grid(self):
        if self.env.states_dim == 1:
            self.maximizers_grid = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)
        elif self.env.states_dim == 2:
            self.maximizers_grid = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))).T.reshape(-1, 2)
        else:
            raise NotImplementedError('Maximizers grid for higher dimensions not implemented yet')

    def ucb_grid_potential_maximizers(self):
        
        lcbs = self.estimator.lcb(torch.tensor(self.maximizers_grid)).numpy().squeeze()
        highest_lcb = np.max(lcbs)

        ucbs = self.estimator.ucb(torch.tensor(self.maximizers_grid)).numpy().squeeze()

        return self.maximizers_grid[ucbs >= highest_lcb]
    
    def step_update(self):
        if self.markovian:
            pass
        else:
            action = self.action_trajectory[-1]
            # see the next state
            next_state = self.env.next(self.env.state, action)
            print('next design: ', next_state)
            # obtain the corresponding observation
            eps = np.random.normal(0, self.sigma**2)
            if callable(self.theta_star):
                fun_value = self.theta_star(next_state)
            else:
                z = self.embedding.embed(torch.tensor(next_state)).numpy()
                eps = np.random.normal(0, self.sigma)
                fun_value = z @ self.theta_star + eps
            
            # add the data point to the queue
            self.queue.append((next_state, fun_value))

            if len(self.queue) == self.asynchronous_delay:
                # add the data point to the estimator
                data_point = self.queue[0]
                self.estimator.add_data_point(torch.tensor(data_point[0]),
                                                torch.tensor([[data_point[1]]]))
                # fit the estimator
                self.estimator.fit()
                # update the maximizers
                if self.maximization_set_method == 'ucb':
                    self.objective.set_of_maximizers = self.ucb_potential_maximizers()
                elif self.maximization_set_method == 'ucb_grid':
                    self.objective.set_of_maximizers = self.ucb_grid_potential_maximizers()    
                elif self.maximization_set_method == 'thompson_sampling':
                    self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
                elif self.maximization_set_method == 'af_mix':
                    self.objective.set_of_maximizers = self.af_mix_potential_maximizers()
                elif self.maximization_set_method == 'none':
                    pass
                # shorten the queue
                self.queue = self.queue[1:]
            
            # keep track of best guess
            if self.keep_track_best_guess:
                best_guess = self.optimum_location_guess()
                self.best_arm.append(best_guess)


    def episode_update(self):
        if self.markovian:
            final_state = self.env.next(self.env.state, self.action_trajectory[-1])
            state_list = self.state_trajectory[1:] + [final_state]
            print('actions taken: ', state_list)

            for state in state_list:
                eps = np.random.normal(0, self.sigma**2)
                if callable(self.theta_star):
                    fun_value = self.theta_star(state) + eps
                else:
                    z = self.embedding.embed(torch.tensor(state)).numpy()
                    fun_value = z @ self.theta_star + eps
                
                self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                        torch.tensor([[fun_value]]))
            
            self.estimator.fit()
            if self.maximization_set_method == 'ucb':
                self.objective.set_of_maximizers = self.ucb_potential_maximizers()
            elif self.maximization_set_method == 'ucb_grid':
                self.objective.set_of_maximizers = self.ucb_grid_potential_maximizers()    
            elif self.maximization_set_method == 'thompson_sampling':
                self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
            elif self.maximization_set_method == 'af_mix':
                self.objective.set_of_maximizers = self.af_mix_potential_maximizers()
            elif self.maximization_set_method == 'none':
                pass

            if self.keep_track_best_guess:
                best_guess = self.optimum_location_guess()
                self.best_arm.append(best_guess)


            if self.video:
                # save the relevant stuff for plotting

                # need to save the state

                # need to save the set of maximizers

                # need to save the state of the GP

                pass
        
        else:
            pass
    
    def optimum_location_guess(self):
        '''
        Estimate the location of the optimum by maximizing the posterior mean
        '''
        # if the estimator is not fitted, return the initial point
        if not self.estimator.fitted:
            return self.initial_point


        # define the function to optimize
        def f(x):
            self.estimator.mean(torch.tensor(x.reshape(1, -1)))
            return -self.estimator.mean(torch.tensor(x.reshape(1, -1))).numpy().squeeze()
        
        # define the bounds
        bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)]

        # initialize the optimizer at the best observed point so far
        x0_idx = np.argmax(self.estimator.y)
        x0 = self.estimator.x[x0_idx, :].numpy()

        # run the optimization
        res = minimize(f, x0 = x0, bounds = bounds, method = 'L-BFGS-B', options = {'maxiter': 1000})

        return res.x.reshape(1, -1)

    def EI(self, x):
        '''
        Expected improvement acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # obtain the best observed value
        best_obs = np.max(self.estimator.y)
        # calculate the improvement
        improvement = mean.numpy().squeeze() - best_obs
        # calculate the expected improvement
        expected_improvement = improvement * norm.cdf(improvement / std.numpy().squeeze()) + std.numpy().squeeze() * norm.pdf(improvement / std.numpy().squeeze())
        # return the negative expected improvement
        return -expected_improvement

    def PI(self, x):
        '''
        Probability of improvement acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # obtain the best observed value
        best_obs = np.max(self.estimator.y)
        # calculate the improvement
        improvement = mean.numpy().squeeze() - best_obs
        # calculate the probability of improvement
        probability_improvement = norm.cdf(improvement / std.numpy().squeeze())
        # return the negative probability of improvement
        return -probability_improvement
    

    
    def UCB(self, x, beta = 2):
        '''
        Upper confidence bound acquisition function
        '''
        # obtain the mean and std of the GP
        mean, std = self.estimator.mean_std(torch.tensor(x.reshape(1, -1)))
        # calculate the upper confidence bound
        ucb = mean.numpy().squeeze() + beta * std.numpy().squeeze()
        # return the negative upper confidence bound
        return -ucb