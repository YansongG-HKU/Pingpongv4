import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import heapq

# Neural network for Deep Q Learning
def create_q_model(num_actions):
    inputs = layers.Input(shape=(210, 160, 3))

    # Convolutions on the frames
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)

# Environment setup
env = gym.make('PongNoFrameskip-v4')
num_actions = env.action_space.n

# The first model makes the predictions for Q-values which are used to make an action.
model = create_q_model(num_actions)
# Build a target model for the prediction of future rewards.
model_target = create_q_model(num_actions)

# Experience replay buffers
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.priority_queue = []
        self.capacity = capacity
        self.counter = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.priority_queue, (self.counter, experience))
            self.buffer.append(experience)
        else:
            _, experience_to_remove = heapq.heappop(self.priority_queue)
            self.buffer.remove(experience_to_remove)
            heapq.heappush(self.priority_queue, (self.counter, experience))
            self.buffer.append(experience)
        self.counter += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

replay_buffer = PrioritizedReplayBuffer(capacity=100000)

# Other training parameters
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000
loss_function = tf.keras.losses.Huber()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00025, rho=0.95, momentum=0.0, epsilon=0.00001, centered=False)

epsilon = 1.0
batch_size = 32
gamma = 0.99  # Discount factor for past rewards

# Training loop
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

while True:  # Run until solved
    state, _ = env.reset()  # Extract only the state part of the reset return value
    state = np.array(state)
    episode_reward = 0

    for timestep in range(1, 1000000):
        frame_count += 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        epsilon -= (1.0 - 0.1) / epsilon_greedy_frames
        epsilon = max(epsilon, 0.1)

        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.array(next_state)
        episode_reward += reward

        replay_buffer.add((state, action, reward, next_state, done))

        state = next_state

        if frame_count % update_after_actions == 0 and len(replay_buffer) > batch_size:
            indices = np.random.choice(range(len(replay_buffer)), size=batch_size)
            experiences = [replay_buffer.buffer[i] for i in indices]

            state_sample = np.array([exp[0] for exp in experiences])
            state_next_sample = np.array([exp[3] for exp in experiences])
            rewards_sample = [exp[2] for exp in experiences]
            action_sample = [exp[1] for exp in experiences]
            done_sample = tf.convert_to_tensor(
                [float(exp[4]) for exp in experiences]
            )

            future_rewards = model_target.predict(state_next_sample)
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        if done:
            break

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]

    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
