import gymnasium as gym 
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Initialize the Q-table globally
q = defaultdict(lambda: np.zeros(gym.make('CartPole-v1').action_space.n))

def run(is_training = True, render = False):
    
    #hyperparameter
    learning_rate_alpha = 0.5
    discount_factor_gamma = 0.99
    epsilon = 1
    epsilon_decay_rate = 0.00001
    #epsilon_min = 0.001
    
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    x     = np.linspace(-2.4, 2.4, 10)
    v     = np.linspace(-4, 4, 10)
    theta = np.linspace(-.2095, .2095, 10)  
    w     = np.linspace(-4, 4, 10)

    
    #Training / Executing

    reward_per_episode = []         # list to store the reward for each episode
    i = 0                           # episode counter

    #While on the episodes
    while(True):

        state = env.reset()[0]

        s_i0 = np.digitize(state[0], x)
        s_i1 = np.digitize(state[1], v)
        s_i2 = np.digitize(state[2], theta)
        s_i3 = np.digitize(state[3], w)

        rewards = 0
        terminated = False

        #While on the SINGLE episode
        while(not terminated and rewards < 10000):
            #Random action
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            #Best Action; key: (s_i0, s_i1, s_i2, s_i3) value: argmax = best action
            else:
                action= np.argmax(q[s_i0, s_i1, s_i2, s_i3])
                
            #Take action
            new_state, reward, terminated, _, _ = env.step(action)

            #Discretize the new states
            ns_i0 = np.digitize(new_state[0], x)
            ns_i1 = np.digitize(new_state[1], v)
            ns_i2 = np.digitize(new_state[2], theta)
            ns_i3 = np.digitize(new_state[3], w)

            if (is_training):
                future_q_value =np.max(q[ns_i0, ns_i1, ns_i2, ns_i3])
                temporal_difference = (
                    reward + discount_factor_gamma * future_q_value - q[s_i0, s_i1, s_i2, s_i3][action]
                    )
                q[s_i0, s_i1, s_i2, s_i3][action] = (
                    q[s_i0, s_i1, s_i2, s_i3][action] + learning_rate_alpha * temporal_difference
                )

            #Set state to new state
            state = new_state

            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2
            s_i3 = ns_i3


            #Collect reward at each step

            rewards += reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i} Rewards: {rewards}')

        
        reward_per_episode.append(rewards)
        mean_rewards = np.mean(reward_per_episode[len(reward_per_episode)-100:])

        if is_training and i%100==0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:
            break

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Increment episode counter
        i+=1   

    env.close()

        
def main():
    run(is_training=True, render=False)
    run(is_training=False, render=True)

main()