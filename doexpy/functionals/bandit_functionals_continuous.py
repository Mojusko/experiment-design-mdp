from doexpy.functionals.reward_functional import ContinuousRewardFunctional


class DesignBestArmLinearBanditNoDenominatorContinuous(ContinuousRewardFunctional):

    def __init__(self, env, lambd, embedding , variant: int = 0, eps=0.01, sigma=0.01, scale_reg=True, num_of_maximizers = 25):

        super().__init__()

        self.env = env
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps
        self.sigma = sigma
        self.uniform_alpha = False
        self.scale_reg = scale_reg
        # initialize the set of maximizers uniformly
        self.num_of_maximizers = num_of_maximizers
        self.set_of_maximizers = np.random.uniform(low = -0.5, high = 0.5, size = (self.num_of_maximizers, self.env.states_dim))

        # define the embedding
        if embedding is None:
            self.embedding = EmptyEmbedding()
        else:
            self.embedding = embedding

    def _estimate_building_blocks(self, emissions, V_eta_inv):

        set_of_maximizers = self.embedding.embed(torch.tensor(self.set_of_maximizers)).numpy()
        n, m = set_of_maximizers.shape
        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = set_of_maximizers.reshape(n, 1, m)
        arr2_reshaped = set_of_maximizers.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)        
        star_id = np.argmax(nominators)
        diff_star = diffs[star_id].reshape(1, -1)

        val_star = np.max(nominators)

        return diffs, diff_star, val_star
    

    def _estimate_diff_sum(self):

        set_of_maximizers = self.embedding.embed(torch.tensor(self.set_of_maximizers)).numpy()
        n, m = set_of_maximizers.shape
        # Reshape the arrays to have compatible shapes for broadcasting

        arr1_reshaped = set_of_maximizers.reshape(n, 1, m)
        arr2_reshaped = set_of_maximizers.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        # now compute outer product
        diffs_outer = np.einsum('ij,ik->ijk', diffs, diffs)
        diffs_outer = diffs_outer.reshape((-1, diffs_outer.shape[-1]))

        # finally compute the sum
        diffs_outer_sum = np.mean(diffs_outer, axis = 0).reshape(1, -1)

        return diffs_outer_sum

    
    def precompute_z_star_and_V_eta_inv(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):
        '''
        Find the action that maximizes the objective for the current distribution, to avoid having to recompute when calculating
        the gradient. Given as:

        z_star = argmax_z z^T V_eta_inv z
        '''

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/(self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0

        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        new_V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta
        
        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(self.embedding.m))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))

        _, diff_star, _ = self._estimate_building_blocks(emissions, V_eta_inv)

        self.z_star = diff_star
        self.precomputed_V_eta_inv = V_eta_inv
    
    def precompute_z_sum_and_V_eta_inv(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):
        '''
        Find the action that maximizes the objective for the current distribution, to avoid having to recompute when calculating
        the gradient. Given as:

        z_star = argmax_z z^T V_eta_inv z
        '''

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/(self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0

        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        new_V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta
        
        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(self.embedding.m))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))

        self.z_star = self._estimate_diff_sum()
        self.precomputed_V_eta_inv = V_eta_inv

    def precompute_agg_V_eta(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if (len(unrolls) > 0) & (len(unrolls[0][1]) > 0):
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/ (self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0
        
        self.precomputed_agg_V_eta = agg_V_eta
        self.precomputed_alpha = alpha

    def pre_compute(self, emissions, distribution, unrolls, episodes):
        self.precompute_z_star_and_V_eta_inv(emissions, distribution, unrolls, episodes)
        # self.precompute_z_sum_and_V_eta_inv(emissions, distribution, unrolls, episodes)
        self.precompute_agg_V_eta(emissions, distribution, unrolls, episodes)

    def get_gradient_density(self, emissions, distribution, unrolls, episodes, x, a):
        '''
        Compute the gradient of the objective with respect to the density, calculated as:

        Trace(z_star z_star^T V_eta_inv phi(x, a) phi(x, a)^T V_eta_inv)
        = Trace(V_eta_inv z_star z_star^T V_eta_inv phi(x, a) phi(x, a)^T)

        x - state (batch_size, state_dim)

        output: (batch_size, 1)

        '''
        # check type for x and a and use numpy or torch
        if isinstance(x, torch.Tensor):

            if isinstance(self.precomputed_V_eta_inv, np.ndarray):
                self.precomputed_V_eta_inv = torch.tensor(self.precomputed_V_eta_inv)
            if isinstance(self.z_star, np.ndarray):
                self.z_star = torch.tensor(self.z_star)

            phi_x = self.embedding.embed(x)
            # compute the first part of the matrix multiplication
            mat_1 = self.precomputed_V_eta_inv @ self.z_star.T @ self.z_star @ self.precomputed_V_eta_inv
            # calculated batched outer product
            mat_2 = torch.bmm(phi_x.unsqueeze(2), phi_x.unsqueeze(1))
            # multiply together
            mat = torch.matmul(mat_1.unsqueeze(0), mat_2)
            # take the trace of each matrix to obtain the gradient
            gradient = torch.diagonal(mat, dim1=1, dim2=2).sum(dim=1, keepdim = True)

        elif isinstance(x, np.ndarray):
            phi_x = self.embedding.embed(torch.tensor(x)).numpy()
            # compute the first part of the matrix multiplication
            mat_1 = self.precomputed_V_eta_inv @ self.z_star.T @ self.z_star @ self.precomputed_V_eta_inv
            # now multiply with phi_x in row-wise fashion to deal with batch-dim
            gradient = np.apply_along_axis(lambda x: np.trace(mat_1 @ x.reshape(1, -1).T @ x.reshape(1, -1)), 1, phi_x)

        return gradient

    def eval(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):
        
        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if (len(unrolls) > 0) & (len(unrolls[0][1]) > 0):
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/ (self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0


        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        new_V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(self.embedding.m))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star

    def eval_quick(self, emission, distribution, unrolls, episodes):
        '''
        Quick evaluation of the function for cyipopt optimization.

        distribution - (H, state_dim)
        '''
        if isinstance(distribution, torch.Tensor):

            if isinstance(self.precomputed_agg_V_eta, np.ndarray):
                self.precomputed_agg_V_eta = torch.tensor(self.precomputed_agg_V_eta)
            
            emissions = self.embedding.embed(distribution)

            new_V_eta = emissions.T @ (torch.eye(distribution.shape[0])/ (self.sigma ** 2)) @ emissions

            if self.uniform_alpha:
                V_eta = 1. / episodes * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta
            else:
                V_eta = (1 - self.precomputed_alpha) * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta

            if not self.scale_reg:
                V_eta_inv = torch.linalg.inv(V_eta + (1 - self.precomputed_alpha) * self.lambd * torch.eye(self.embedding.m))
            else:
                V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(self.embedding.m))
            
            # estimate with building blocks using numpy
            with torch.no_grad():
                # create numpy copy of V_eta_inv
                V_eta_inv_numpy = V_eta_inv.clone().numpy()
                _, diff_star, val_star = self._estimate_building_blocks(None, V_eta_inv_numpy)
                diff_star = torch.tensor(diff_star).reshape(-1, 1)

            # now re-calculate val_star using torch to be able to backprop
            val_star = diff_star.T @ V_eta_inv @ diff_star

            return - val_star

        elif isinstance(distribution, np.ndarray):

            if isinstance(self.precomputed_agg_V_eta, torch.Tensor):
                self.precomputed_agg_V_eta = np.array(self.precomputed_agg_V_eta)
            
            emissions = self.embedding.embed(torch.tensor(distribution)).numpy()

            new_V_eta = emissions.T @ (np.eye(distribution.shape[0])/ (self.sigma ** 2)) @ emissions

            if self.uniform_alpha:
                V_eta = 1. / episodes * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta
            else:
                V_eta = (1 - self.precomputed_alpha) * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta

            if not self.scale_reg:
                V_eta_inv = np.linalg.inv(V_eta + (1 - self.precomputed_alpha) * self.lambd * np.identity(self.embedding.m))
            else:
                V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))
            
            _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        
        return - val_star

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        
        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))
        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star





class G_DesignBestArmLinearBanditNoDenominatorContinuous(ContinuousRewardFunctional):

    def __init__(self, env, lambd, embedding , variant: int = 0, eps=0.01, sigma=0.01, scale_reg=True, num_of_maximizers = 25):

        super().__init__()

        self.env = env
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps
        self.sigma = sigma
        self.uniform_alpha = False
        self.scale_reg = scale_reg
        # initialize the set of maximizers uniformly
        self.num_of_maximizers = num_of_maximizers
        self.set_of_maximizers = np.random.uniform(low = -0.5, high = 0.5, size = (self.num_of_maximizers, self.env.states_dim))

        # define the embedding
        if embedding is None:
            self.embedding = EmptyEmbedding()
        else:
            self.embedding = embedding

    def _estimate_building_blocks(self, emissions, V_eta_inv):

        set_of_maximizers = self.embedding.embed(torch.tensor(self.set_of_maximizers)).numpy()

        nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, set_of_maximizers)        
        star_id = np.argmax(nominators)
        z_star = set_of_maximizers[star_id].reshape(1, -1)

        val_star = np.max(nominators)

        return None, z_star, val_star

    
    def precompute_z_star_and_V_eta_inv(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):
        '''
        Find the action that maximizes the objective for the current distribution, to avoid having to recompute when calculating
        the gradient. Given as:

        z_star = argmax_z z^T V_eta_inv z
        '''

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/(self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0

        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        new_V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta
        
        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(self.embedding.m))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))

        _, z_star, _ = self._estimate_building_blocks(emissions, V_eta_inv)

        self.z_star = z_star
        self.precomputed_V_eta_inv = V_eta_inv
    

    def precompute_agg_V_eta(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if (len(unrolls) > 0) & (len(unrolls[0][1]) > 0):
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/ (self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0
        
        self.precomputed_agg_V_eta = agg_V_eta
        self.precomputed_alpha = alpha

    def pre_compute(self, emissions, distribution, unrolls, episodes):
        self.precompute_z_star_and_V_eta_inv(emissions, distribution, unrolls, episodes)
        # self.precompute_z_sum_and_V_eta_inv(emissions, distribution, unrolls, episodes)
        self.precompute_agg_V_eta(emissions, distribution, unrolls, episodes)

    def get_gradient_density(self, emissions, distribution, unrolls, episodes, x, a):
        '''
        Compute the gradient of the objective with respect to the density, calculated as:

        Trace(z_star z_star^T V_eta_inv phi(x, a) phi(x, a)^T V_eta_inv)
        = Trace(V_eta_inv z_star z_star^T V_eta_inv phi(x, a) phi(x, a)^T)

        x - state (batch_size, state_dim)

        output: (batch_size, 1)

        '''
        # check type for x and a and use numpy or torch
        if isinstance(x, torch.Tensor):

            if isinstance(self.precomputed_V_eta_inv, np.ndarray):
                self.precomputed_V_eta_inv = torch.tensor(self.precomputed_V_eta_inv)
            if isinstance(self.z_star, np.ndarray):
                self.z_star = torch.tensor(self.z_star)

            phi_x = self.embedding.embed(x)
            # compute the first part of the matrix multiplication
            mat_1 = self.precomputed_V_eta_inv @ self.z_star.T @ self.z_star @ self.precomputed_V_eta_inv
            # calculated batched outer product
            mat_2 = torch.bmm(phi_x.unsqueeze(2), phi_x.unsqueeze(1))
            # multiply together
            mat = torch.matmul(mat_1.unsqueeze(0), mat_2)
            # take the trace of each matrix to obtain the gradient
            gradient = torch.diagonal(mat, dim1=1, dim2=2).sum(dim=1, keepdim = True)

        elif isinstance(x, np.ndarray):
            phi_x = self.embedding.embed(torch.tensor(x)).numpy()
            # compute the first part of the matrix multiplication
            mat_1 = self.precomputed_V_eta_inv @ self.z_star.T @ self.z_star @ self.precomputed_V_eta_inv
            # now multiply with phi_x in row-wise fashion to deal with batch-dim
            gradient = np.apply_along_axis(lambda x: np.trace(mat_1 @ x.reshape(1, -1).T @ x.reshape(1, -1)), 1, phi_x)

        return gradient

    def eval(self, emissions, distribution: NonStationaryDeltaDensity, unrolls, episodes):
        
        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        if (len(unrolls) > 0) & (len(unrolls[0][1]) > 0):
            aggregated_density = self.build_density_from_trajectories(unrolls)
            aggregated_density = aggregated_density.average_density()
            aggregated_emissions = self.embedding.embed(torch.tensor(aggregated_density.delta_states)).numpy()
            aggregated_weights = aggregated_density.weights

            agg_V_eta = aggregated_emissions.T @ (np.diag(aggregated_weights)/ (self.sigma ** 2)) @ aggregated_emissions

        else:
            aggregated_density = 0
            aggregated_emissions = 0
            aggregated_weights = 0

            agg_V_eta = 0


        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        new_V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(self.embedding.m))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star

    def eval_quick(self, emission, distribution, unrolls, episodes):
        '''
        Quick evaluation of the function for cyipopt optimization.

        distribution - (H, state_dim)
        '''
        if isinstance(distribution, torch.Tensor):

            if isinstance(self.precomputed_agg_V_eta, np.ndarray):
                self.precomputed_agg_V_eta = torch.tensor(self.precomputed_agg_V_eta)
            
            emissions = self.embedding.embed(distribution)

            new_V_eta = emissions.T @ (torch.eye(distribution.shape[0])/ (self.sigma ** 2)) @ emissions

            if self.uniform_alpha:
                V_eta = 1. / episodes * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta
            else:
                V_eta = (1 - self.precomputed_alpha) * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta

            if not self.scale_reg:
                V_eta_inv = torch.linalg.inv(V_eta + (1 - self.precomputed_alpha) * self.lambd * torch.eye(self.embedding.m))
            else:
                V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(self.embedding.m))
            
            # estimate with building blocks using numpy
            with torch.no_grad():
                # create numpy copy of V_eta_inv
                V_eta_inv_numpy = V_eta_inv.clone().numpy()
                _, diff_star, val_star = self._estimate_building_blocks(None, V_eta_inv_numpy)
                diff_star = torch.tensor(diff_star).reshape(-1, 1)

            # now re-calculate val_star using torch to be able to backprop
            val_star = diff_star.T @ V_eta_inv @ diff_star

            return - val_star

        elif isinstance(distribution, np.ndarray):

            if isinstance(self.precomputed_agg_V_eta, torch.Tensor):
                self.precomputed_agg_V_eta = np.array(self.precomputed_agg_V_eta)
            
            emissions = self.embedding.embed(torch.tensor(distribution)).numpy()

            new_V_eta = emissions.T @ (np.eye(distribution.shape[0])/ (self.sigma ** 2)) @ emissions

            if self.uniform_alpha:
                V_eta = 1. / episodes * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta
            else:
                V_eta = (1 - self.precomputed_alpha) * new_V_eta + \
                    self.precomputed_alpha * self.precomputed_agg_V_eta

            if not self.scale_reg:
                V_eta_inv = np.linalg.inv(V_eta + (1 - self.precomputed_alpha) * self.lambd * np.identity(self.embedding.m))
            else:
                V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))
            
            _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        
        return - val_star

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        
        average_distribution = distribution.average_density()
        emissions = self.embedding.embed(torch.tensor(average_distribution.delta_states)).numpy()
        weights = average_distribution.weights

        V_eta = emissions.T @ (np.diag(weights)/ (self.sigma ** 2)) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(self.embedding.m))
        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star