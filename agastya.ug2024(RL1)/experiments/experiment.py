import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from utils.features import RadialBasisFunctions, TileCoder
from agents.algorithms import Sarsa, QLearning
from utils.runner import train_agent

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    
    min_feat = env.observation_space.low
    max_feat = env.observation_space.high
    
    rbf_extractor = RadialBasisFunctions(
        min_features=min_feat, max_features=max_feat, 
        num_centres=10, norm=2, sigma=0.1
    )
    
    alpha = 0.1 / rbf_extractor.feature_dim
    gamma = 0.99
    epsilon_start = 1.0  
    episodes = 400
    
    sarsa_agent = Sarsa(env.action_space.n, rbf_extractor, alpha, gamma, epsilon_start)
    q_agent = QLearning(env.action_space.n, rbf_extractor, alpha, gamma, epsilon_start)
    
    print("--- Training SARSA ---")
    sarsa_returns = train_agent(env, sarsa_agent, episodes, epsilon_decay=0.995)
    
    print("\n--- Training Q-Learning ---")
    q_returns = train_agent(env, q_agent, episodes, epsilon_decay=0.995)
    
    env.close()

    plt.figure(figsize=(10, 6))
    window = 15
    plt.plot(np.convolve(sarsa_returns, np.ones(window)/window, mode='valid'), label='SARSA', color='blue')
    plt.plot(np.convolve(q_returns, np.ones(window)/window, mode='valid'), label='Q-Learning', color='red', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Return (Smoothed)')
    plt.title('Mountain Car: SARSA vs Q-Learning (Epsilon Decay)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n=== Watching the Trained Agents in Action ===")
    render_env = gym.make("MountainCar-v0", render_mode="human")
    
    print("Rendering SARSA...")
    sarsa_agent.epsilon = 0.0  
    state, _ = render_env.reset()
    done = False
    while not done:
        action = sarsa_agent.act(state)
        state, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated

    print("Rendering Q-Learning...")
    q_agent.epsilon = 0.0  
    state, _ = render_env.reset()
    done = False
    while not done:
        action = q_agent.act(state)
        state, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        
    render_env.close()