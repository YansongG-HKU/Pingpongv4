import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Define a function to preprocess the state (resize and grayscale)
def preprocess_state(state_image):
    state_image = np.array(state_image, dtype=np.uint8)  # Ensure the image is a numpy array of type uint8
    state_image = tf.image.rgb_to_grayscale(state_image)  # Convert to grayscale
    state_image = tf.image.crop_to_bounding_box(state_image, 35, 0, 160, 160)  # Crop the image
    state_image = tf.image.resize(state_image, [84, 84])  # Resize to 84x84
    state_image = state_image / 255.0  # Normalize pixel values to [0, 1]
    return state_image.numpy()  # Convert back to numpy array

# Create environment
env = gym.make('PongDeterministic-v4', render_mode='human')

# Define the Q-network model
model = keras.Sequential([
    keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)),
    keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(env.action_space.n)
])

# Define the target Q-network model
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

# Define optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss_fn = keras.losses.MeanSquaredError()

# Hyperparameters
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
batch_size = 32

# Initialize replay buffer
replay_buffer = []

# Initialize state
state = env.reset()

done = False
n_steps = 0

# Game loop
while not done:
    # Epsilon-greedy exploration
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        preprocessed_state = preprocess_state(state)
        q_values = model.predict(np.array([preprocessed_state]))[0]
        action = np.argmax(q_values)

    # Perform actions and get new states, rewards, and game completion status
    observation = env.step(action)
    next_state, reward, done, _, info = observation

    # Store the transition in the replay buffer
    replay_buffer.append((state, action, reward, next_state, done))

    # Limit the replay buffer size
    if len(replay_buffer) > 10000:
        replay_buffer.pop(0)

    # Sample a random batch from the replay buffer
    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)

        # Calculate target Q-values
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.stack([preprocess_state(s) for s, _, _, _, _ in batch])
        next_states = np.stack([preprocess_state(s) for _, _, _, s, _ in batch])

        q_values = model.predict(states)
        next_q_values = target_model.predict(next_states)
        targets = q_values.copy()

        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

        # Update the Q-network
        with tf.GradientTape() as tape:
            q_values = model(states, training=True)
            loss = loss_fn(targets, q_values)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update the target Q-network periodically
    if n_steps % 1000 == 0:
        target_model.set_weights(model.get_weights())

    # Update epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Update state and step count
    state = next_state
    n_steps += 1

print(f"Episode finished after {n_steps} steps")
