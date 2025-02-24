import gymnasium as gym 
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Initialize the Q-table globally
q = None
q_saved = None  # Variable to store Q-table from the first run with specific hyperparameters

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
        plt.savefig(f"/home/dans/Projects/ML/src/Results/Tabular/fig_{label.replace(', ', '_').replace('=', '')}.png")
        plt.close()

def run(is_training=True, render=False, 
        learning_rate_alpha=0.2, 
        discount_factor_gamma=0.99, 
        epsilon=1, 
        epsilon_decay_rate=0.00001, 
        n_episodes=100000):

    global q, q_saved

    # Reset the Q-table for every run
    q = defaultdict(lambda: np.zeros(gym.make('CartPole-v1').action_space.n))

    # Load the saved Q-table for the specific case
    if not is_training and q_saved is not None:
        q = q_saved

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    #Creation of the intervals  

    x     = np.linspace(-2.4, 2.4, 10)
    v     = np.linspace(-4, 4, 10)
    theta = np.linspace(-.2095, .2095, 10)  
    w     = np.linspace(-4, 4, 10)

    reward_to_plot = []
    reward_per_episode = []         # list to store the reward for each episode
    i = 0                           # episode counter

    while(i < n_episodes):

        state = env.reset()[0]

        # Discretization of the states
        s_i0 = np.digitize(state[0], x)
        s_i1 = np.digitize(state[1], v)
        s_i2 = np.digitize(state[2], theta)
        s_i3 = np.digitize(state[3], w)

        rewards = 0
        terminated = False

        #While on the single episode
        while(not terminated and rewards < 10000):
            #Takes a random action or the best action previously saved
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action= np.argmax(q[s_i0, s_i1, s_i2, s_i3])

            new_state, reward, terminated, _, _ = env.step(action)

            ns_i0 = np.digitize(new_state[0], x)
            ns_i1 = np.digitize(new_state[1], v)
            ns_i2 = np.digitize(new_state[2], theta)
            ns_i3 = np.digitize(new_state[3], w)


            ## If the agent is training , update the Q table with the Bellman Equation 
            if (is_training):
                future_q_value =np.max(q[ns_i0, ns_i1, ns_i2, ns_i3])
                temporal_difference = (
                    reward + discount_factor_gamma * future_q_value - q[s_i0, s_i1, s_i2, s_i3][action]
                    )
                q[s_i0, s_i1, s_i2, s_i3][action] = (
                    q[s_i0, s_i1, s_i2, s_i3][action] + learning_rate_alpha * temporal_difference
                )

            state = new_state
            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2
            s_i3 = ns_i3

            rewards += reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i} Rewards: {rewards}')


        #Useful for plotting =)
        reward_per_episode.append(rewards)
        mean_rewards = np.mean(reward_per_episode[len(reward_per_episode)-100:])
        reward_to_plot.append(mean_rewards)

        #Log
        if is_training and i%100==0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        #break condition
        if mean_rewards > 1000:
            break
        
        #epsilon decay
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i+=1   

    env.close()

    # Save rewards for plotting
    label = f"alpha={learning_rate_alpha}, gamma={discount_factor_gamma}, epsilon_decay={epsilon_decay_rate}"
    plot_data[label] = reward_to_plot

    # Save the Q-table if specific conditions are met
    if is_training and learning_rate_alpha == 0.2 and discount_factor_gamma == 0.99 and epsilon_decay_rate == 0.00001:
        q_saved = q.copy()

def main():
    #Correct Run
    run(is_training=True, render=False, 
        learning_rate_alpha = 0.2,
        discount_factor_gamma = 0.99,
        epsilon = 1,
        epsilon_decay_rate = 0.00001,
        n_episodes = 100000)
    
    run(is_training=True, render=False, 
        learning_rate_alpha = 0.2,
        discount_factor_gamma = 0.7,
        epsilon = 1,
        epsilon_decay_rate = 0.00001,
        n_episodes = 100000)

    run(is_training=True, render=False, 
        learning_rate_alpha = 0.2,
        discount_factor_gamma = 0.7,
        epsilon = 1,
        epsilon_decay_rate = 0.0001,
        n_episodes = 100000)
    
    run(is_training=True, render=False, 
        learning_rate_alpha = 0.2,
        discount_factor_gamma = 0.5,
        epsilon = 1,
        epsilon_decay_rate = 0.00001,
        n_episodes = 100000)

    plot_results()
    
    #run(is_training=False, render=True)

main()
