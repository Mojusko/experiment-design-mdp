import numpy as np
from mdpexplore.solvers.dp import DP
from mdpexplore.env.discrete_env import DiscreteEnv

class DummyTestEnv(DiscreteEnv):
    def __init__(self, states_num = 5, actions_num = 5, max_episode_length = 10, constrained = False):
        super().__init__(init_state=0)
        self.states_num = states_num
        self.actions_num = actions_num
        self.max_episode_length = max_episode_length
        self.terminal_state = None
        self.constrained = constrained

    def available_actions(self, state):
        return [a for a in range(self.actions_num) if self.is_valid_action(a, state)]

    def next(self, state, action):
        return action
    
    def convert(self, state):
        pass

    def step(self, action: int):
        self.visitations[action] += 1
        return action

    def get_transition_matrix(self) -> np.ndarray:
        tm = np.zeros((self.states_num, self.actions_num, self.states_num))
        for s in range(self.states_num):
            for a in range(self.states_num):
                if self.is_valid_action(a, s):
                    tm[s, a, a] = 1.
        return tm

    def is_valid_action(self, action, state) -> bool:
        if state == 0:
            return action in [0, 1]
            
        if state == 1:
            return action in [0, 1, 2]

        
        if state == 2:
            return action in [1, 2, 3]
        
        if state == 3:
            return action in [2, 3, 4]
        
        if state == 4:
            return action in [3, 4]

    def reset(self) -> None:
        self.state = self.init_state

# initialize reward
reward = np.zeros((10, 5, 5))
# reward for staying in the rightmost state
reward[:, 4, 4] = 1.0
# reward for transitioning to the right
for i in range(4):
    reward[:, i, i+1] = 0.1

env = DummyTestEnv()
solver = DP(env, reward)
policy = solver.solve()

def test_policy_is_valid():
    for h in range(10):
        for s in range(5):
            for a in range(5):
                if policy.ps[h, s, a] > 0:
                    assert env.is_valid_action(a, s), 'policy breaks environment constraints'
                    assert policy.ps[h, s, a] <= 1, 'policy suggests a probability greater than 1'
                else:
                    assert policy.ps[h, s, a] == 0, 'policy suggests a negative probability'

def test_policy_is_optimal():
    for i in range(10):
        assert np.all( policy.ps[i] == np.array([[0., 1., 0., 0., 0.],
                                                [0., 0., 1., 0., 0.],
                                                [0., 0., 0., 1., 0.],
                                                [0., 0., 0., 0., 1.],
                                                [0., 0., 0., 0., 1.]])), 'policy is not optimal'