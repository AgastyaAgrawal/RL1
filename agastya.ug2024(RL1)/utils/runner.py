from tqdm import tqdm 

def run_episode(env, agent):
    state, _ = env.reset() #We will use the dictionary if we want to create a cheater agent, which is also important. 
    action = agent.act(state) #check, maybe need to change this. If fine, delete this comment. 
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated): 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_action = agent.act(next_state)
        agent.update(state, action, reward, next_state, next_action, terminated) # we do not want the reward misspecification from the time limit problem. 
        state = next_state
        action = next_action
        total_reward += reward
    
    return total_reward

def train_agent(env, agent, num_episodes, epsilon_decay=0.99, min_epsilon=0.01):
    returns = []
    progress_bar = tqdm(range(num_episodes), desc = "Training Episodes")
    for ep in progress_bar:
        ep_return = run_episode(env, agent)
        returns.append(ep_return)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(min_epsilon, agent.epsilon*epsilon_decay)
            progress_bar.set_postfix({'Return': ep_return, 'Eps': f"{agent.epsilon:.2f}"})
        else:
            progress_bar.set_postfix({'Return': ep_return})
    return returns

def run_mc_episode(env, agent):
    state, _ = env.reset()
    trajectory = []
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        trajectory.append((state, action, reward)) #just store the entire sequence. 
        
        state = next_state
        total_reward += reward
        
    return trajectory, total_reward

def train_mc_agent(env, agent, num_episodes, epsilon_decay=1.0, min_epsilon=0.01):
    returns = []
    progress_bar = tqdm(range(num_episodes), desc="MC Training Episodes")
    
    for ep in progress_bar:
        trajectory, ep_return = run_mc_episode(env, agent)
        returns.append(ep_return)
        agent.update(trajectory)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
            progress_bar.set_postfix({'Return': ep_return, 'Eps': f"{agent.epsilon:.2f}"})
        else:
            progress_bar.set_postfix({'Return': ep_return})
            
    return returns