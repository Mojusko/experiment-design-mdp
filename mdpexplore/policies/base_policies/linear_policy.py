from mdpexplore.policies.policy_base import Policy

class LinearPolicy(Policy):

    def __init__(self, env, K):
        super().__init__(env)
        self.env = env
        self.K = K

    def next_action(self, state):
        return self.K @ state
