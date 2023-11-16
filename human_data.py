import gym
import numpy as np
import tkinter as tk

# Create the environment
env = gym.make('PongDeterministic-v4', render_mode='human')

# Initialize the state
state, _ = env.reset()

# Create a tkinter window to capture key presses
window = tk.Tk()
window.title("Pong Control")

# Variable to store the current action
current_action = 0  # No-op

# Lists to store states, actions, rewards, next states, and done flags
states = []
actions = []
rewards = []
next_states = []
dones = []

# Function to update the action based on key presses
def key_pressed(event):
    global current_action
    if event.keysym == 'Up':
        current_action = 2  # 'Up' action in Pong
    elif event.keysym == 'Down':
        current_action = 3  # 'Down' action in Pong
    elif event.keysym == 'q':
        window.quit()  # Close the window

# Function to reset the action when a key is released
def key_released(event):
    global current_action
    if event.keysym in ['Up', 'Down']:
        current_action = 0  # No-op

# Bind key events to the tkinter window
window.bind('<KeyPress>', key_pressed)
window.bind('<KeyRelease>', key_released)

# Function to run the game loop
def game_loop():
    global state, current_action

    # Perform the action and get the new state, reward, done flag, etc.
    next_state, reward, done, truncated, info = env.step(current_action)

    # Add the data to the corresponding lists
    states.append(state)
    actions.append(current_action)
    rewards.append(reward)
    next_states.append(next_state)
    dones.append(done)

    # Update the current state
    state = next_state

    # Render the game screen
    env.render()

    if not done:
        window.after(100, game_loop)
    else:
        env.close()

# Start the game loop
window.after(100, game_loop)
window.mainloop()

# Save the data
np.save('states.npy', np.array(states))
np.save('actions.npy', np.array(actions))
np.save('rewards.npy', np.array(rewards))
np.save('next_states.npy', np.array(next_states))
np.save('dones.npy', np.array(dones))
print("Human player data has been saved.")
