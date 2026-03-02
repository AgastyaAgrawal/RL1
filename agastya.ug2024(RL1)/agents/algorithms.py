import numpy as np 
from agents.learning import LinearValueFunction

class Sarsa: 

    def __init__(self, num_actions: int, feature_extractor, learning_rate: float, gamma: float, epsilon: float):
        self.num_actions = num_actions
        self.extractor = feature_extractor
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = LinearValueFunction(num_actions=self.num_actions, feature_dim=self.extractor.feature_dim)

    def act(self, state: np.ndarray) -> int:
        features = self.extractor.extract(state)
        q_values = self.value_function.predict(features)

        if np.random.rand() < self.epsilon: # we love epsilon greedy. 
            return np.random.randint(self.num_actions)
        else:
            return int(np.argmax(q_values))
        
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: int, terminated: bool = False):
        features = self.extractor.extract(state)
        next_features = self.extractor.extract(next_state)

        q_values = self.value_function.predict(features)
        next_q_values = self.value_function.predict(next_features)

        q_current = q_values[action]
        q_next = next_q_values[next_action]
        
        delta = reward + (0 if terminated else 1)*(self.gamma*q_next) - q_current #target - current
        self.value_function.update(features, action, delta, self.learning_rate)

class QLearning: 

    def __init__(self, num_actions: int, feature_extractor, learning_rate: float, gamma: float, epsilon: float):
        self.num_actions = num_actions
        self.extractor = feature_extractor
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon 
        self.value_function = LinearValueFunction(feature_dim=self.extractor.feature_dim, num_actions=self.num_actions)

    def act(self, state: np.ndarray) -> int:
        features = self.extractor.extract(state)
        q_values = self.value_function.predict(features)

        if np.random.rand() < self.epsilon: 
            return np.random.randint(self.num_actions)
        else:
            return int(np.argmax(q_values))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: int = None, terminated: bool = False):
        # The None is for nicely changing from Sarsa to Q_learnign without changing the entire code in experiments.py
        features = self.extractor.extract(state)
        next_features = self.extractor.extract(next_state)

        q_values = self.value_function.predict(features)
        next_q_values = self.value_function.predict(next_features)

        q_current = q_values[action]
        q_next_max = np.max(next_q_values)

        delta = reward + (0 if terminated else 1)*(self.gamma*q_next_max) - q_current
        self.value_function.update(features,action,delta, self.learning_rate)
    
from agents.learning import TabularValueFunction

#The key difference is to use the TabularValueFunction insteaf of the feature extractor. 
class TabularSarsa:

    def __init__(self, num_states, num_actions, alpha, gamma, epsilon):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = TabularValueFunction(num_states, num_actions)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.value_function.predict(state)
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, terminated):
        q_current = self.value_function.predict(state)[action]
        q_next = self.value_function.predict(next_state)[next_action]    
        delta = reward + (0 if terminated else 1)*self.gamma * q_next - q_current
        self.value_function.update(state, action, delta, self.alpha)

class MonteCarloControl:
    def __init__(self, num_states, num_actions, gamma, epsilon):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = TabularValueFunction(num_states, num_actions)
        self.returns_count = np.zeros((num_states, num_actions))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.value_function.predict(state)
        return np.argmax(q_values)

    def update(self, trajectory):
        G = 0
        returns_at_t = []
        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns_at_t.insert(0, G)
            
        visited_state_actions = set()
        for t, (state, action, reward) in enumerate(trajectory):
            state_action_pair = (state, action)
            
            if state_action_pair not in visited_state_actions:
                visited_state_actions.add(state_action_pair)
                G_t = returns_at_t[t]
                self.returns_count[state, action] += 1
                alpha = 1.0 / self.returns_count[state, action]
                q_current = self.value_function.predict(state)[action]
                delta = G_t - q_current
                self.value_function.update(state, action, delta, alpha)

class TD0Prediction:

    def __init__(self, num_states, num_actions, alpha, gamma):

        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.v_table = np.zeros(num_states) 

    def act(self, state):

        return np.random.randint(self.num_actions)

    def update(self, state, action, reward, next_state, next_action, terminated):

        v_current = self.v_table[state]
        if terminated:
            v_next = 0.0
        else:
            v_next = self.v_table[next_state]
        delta = reward + self.gamma * v_next - v_current #1 step. 
        self.v_table[state] += self.alpha * delta

class FirstVisitMCPrediction:
    def __init__(self, num_states, num_actions, gamma):
        self.num_actions = num_actions
        self.gamma = gamma
        self.v_table = np.zeros(num_states)
        self.returns_count = np.zeros(num_states)

    def act(self, state):
        return np.random.randint(self.num_actions)

    def update(self, trajectory):
        G = 0
        returns_at_t = []

        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns_at_t.insert(0, G)
            
        visited_states = set()

        for t, (state, action, reward) in enumerate(trajectory):
            if state not in visited_states:
                visited_states.add(state)
                G_t = returns_at_t[t]
                self.returns_count[state] += 1
                alpha = 1.0 / self.returns_count[state]
                delta = G_t - self.v_table[state]
                self.v_table[state] += alpha * delta

