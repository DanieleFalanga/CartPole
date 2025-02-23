import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

POLICY_DIR = "/home/dans/Projects/ML/src/q_net.pth"

# Global dictionary to hold rewards for plotting
plot_data = {}

def plot_results():
    plt.figure(figsize=(20, 8))
    plt.ylim(0, 1000)  # Set Y-axis scale

    for label, data in plot_data.items():
        plt.figure(figsize=(20, 8))
        episodes = range(0, len(data) * 100, 100)
        plt.plot(episodes, data, label=label)
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title(f"CartPole Tabular Q learning - {label}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"/home/dans/Projects/ML/src/Results/DQN/fig_{label.replace(', ', '_').replace('=', '')}.png")
        plt.close()

# Neural network class
class DQN(nn.Module): 
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Forward function with its activation function (Relu) 
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay buffer class that implements useful functions for its management
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN_Agent():
    def __init__(self, 
            env,
            is_training, batch_size, 
            learning_rate_alpha, 
            discount_factor_gamma, 
            epsilon, 
            epsilon_decay_rate, 
            n_episodes):
        
        self.env = env
        self.is_training = is_training
        self.batch_size = batch_size
        self.learning_rate_alpha = learning_rate_alpha
        self.discount_factor_gamma = discount_factor_gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.n_episodes = n_episodes
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        if not is_training: 
            self.q_net.load_state_dict(torch.load(POLICY_DIR, map_location=self.device, weights_only=True))
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate_alpha)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    # Take action function that follows the epsilon greedy policy    
    def take_action(self, state):
        if self.is_training and np.random.rand() < self.epsilon:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            return action
        else:
            action = self.q_net(state).max(1).indices.view(1, 1)
            return action
            
    # Main function, where the neural network training occurs
    def optimize(self):
        # If it hasn't made enough useful transitions for training, don't execute it
        if len(self.memory) < self.batch_size:
            return
        
        # Sample the batch from the replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Identify non-terminal next states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        # Calculate the q values predicted by the network
        state_action_values = self.q_net(state_batch).gather(1, action_batch)
        
        # Calculate Q values using Bellman 
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.q_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.discount_factor_gamma) + reward_batch
        
        # Calculate the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the network parameters
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip the gradients to avoid too high values 
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def epsilon_decay_function(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)


# Run function 
def run(is_training=True, render=False, 
        learning_rate_alpha=1e-4, 
        discount_factor_gamma=0.99, 
        epsilon=1, 
        epsilon_decay_rate=0.0001, 
        n_episodes=100000,
        batch_size = 128):
    
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Agent initialization
    agent = DQN_Agent(env, is_training=is_training, batch_size= batch_size, 
            learning_rate_alpha=learning_rate_alpha, 
            discount_factor_gamma=discount_factor_gamma, 
            epsilon=epsilon, 
            epsilon_decay_rate=epsilon_decay_rate, 
            n_episodes=n_episodes)
    
    reward_to_plot = []
    reward_per_episode = []         # list to store the reward for each episode    
    i = 0
    
    # While loop on the number of episodes
    while(i < n_episodes):
        
        state= env.reset()[0]
        # Convert the state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        terminated = False
        rewards = 0

        # While loop on the number of episodes
        while(not terminated and rewards < 10000):
            
            action = agent.take_action(state)
            observation, reward, terminated, _, _ = env.step(action.item())
            reward = torch.tensor([reward], device=agent.device)
            
            if terminated: 
                new_state = None
            else:
                # Convert the state to tensor
                new_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

            if (is_training):
                # Save transition to replay buffer
                agent.memory.push(state, action, new_state, reward)
                agent.optimize()

            # Update to the new state
            state = new_state

            rewards += reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i} Rewards: {rewards}')
        
        # Plot functions
        reward_per_episode.append(rewards)
        mean_rewards = np.mean(reward_per_episode[len(reward_per_episode)-100:])
        reward_to_plot.append(mean_rewards)

        if is_training and i%100==0:
            print(f'Episode: {i} {rewards}  Epsilon: {agent.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        agent.epsilon_decay_function()
        i += 1

    env.close()

    # Save rewards for plotting
    label = f"alpha={learning_rate_alpha}, gamma={discount_factor_gamma}, epsilon_decay={epsilon_decay_rate}"
    plot_data[label] = reward_to_plot
    
if __name__ == "__main__":
    run()