import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Create environment
env = gym.make('PongDeterministic-v4', render_mode='human')

# Initialize state
state = env.reset()

done = False
n_steps = 0

# Game loop
while not done:
    # Here we randomly select an action
    action = env.action_space.sample()

    # Perform actions and get new states, rewards, and game completion status
    observation = ()
    state, reward, done, truncated, info = env.step(action)

    # Print the contents of the observation
    print(f"State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")

    # Render the game screen in 'human' mode
    env.render()

    n_steps += 1

print(f"Episode finished after {n_steps} steps")





