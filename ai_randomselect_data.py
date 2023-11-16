import gym
import numpy as np
import os

# Create the environment
env = gym.make('PongDeterministic-v4', render_mode='human')

# Path to files containing winning game data
files = ['winning_states.npy', 'winning_actions.npy', 'winning_rewards.npy', 'winning_next_states.npy', 'winning_dones.npy']

# Check if existing winning game data exists and load it
if all(os.path.exists(f) for f in files):
    states = list(np.load(files[0], allow_pickle=True))
    actions = list(np.load(files[1], allow_pickle=True))
    rewards = list(np.load(files[2], allow_pickle=True))
    next_states = list(np.load(files[3], allow_pickle=True))
    dones = list(np.load(files[4], allow_pickle=True))
else:
    states, actions, rewards, next_states, dones = [], [], [], [], []

# Number of current winning game data samples
current_wins = len(rewards)

# Loop until the total number of winning game data reaches N
while current_wins < 11:
    # Initialize the state
    state = env.reset()
    done = False

    while not done:
        # Randomly choose an action
        action = env.action_space.sample()

        # Perform the action and get the new state, reward, and game completion status
        next_state, reward, done, truncated, info = env.step(action)

        # Check if a point has been scored (winning condition)
        if reward == 1:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            current_wins += 1  # Update the count of winning game data

        # Update the state
        state = next_state

        # Render the game screen
        env.render()

        # If enough winning game data has been collected, exit the loop
        if current_wins >= 100:
            env.close()  # Close the environment
            break

# After the game ends, save the data
for data, file in zip([states, actions, rewards, next_states, dones], files):
    np.save(file, np.array(data))

print("Winning game data has been saved.")
