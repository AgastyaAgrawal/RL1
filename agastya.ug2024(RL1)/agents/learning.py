import numpy as np

class LinearValueFunction: 
    def __init__(self, feature_dim: int, num_actions: int):
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.weights = np.zeros((num_actions, feature_dim))
        '''shape = (num_actions, feature_dim)'''

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features
    # very weirdly, this is true dependency injection. Delta depends on the algorithms.py file.
    def update(self, features: np.ndarray, action: int, delta: float, alpha: float):
        self.weights[action] += alpha*delta*features

class TabularValueFunction:
    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def predict(self, state: int) -> np.ndarray:
        return self.q_table[state]

    def update(self, state: int, action: int, delta: float, alpha: float):
        self.q_table[state, action] += alpha * delta