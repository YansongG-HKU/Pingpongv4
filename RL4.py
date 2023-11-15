import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

# 预处理游戏帧的函数
def preprocess_frame(frame):
    gray = np.mean(frame, axis=2).astype(np.uint8)  # 转换为灰度图
    gray_expanded = np.expand_dims(gray, axis=-1)  # 增加一个维度
    resized = tf.image.resize(gray_expanded, [80, 80])  # 调整大小为 80x80
    return np.array(resized).reshape(80, 80, 1)  # 重塑为 (80, 80, 1)

# 定义 DQN 代理类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 创建和初始化环境
env = gym.make('PongDeterministic-v4')
state_size = (80, 80, 1)
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32

# 游戏循环
state = preprocess_frame(env.reset())
state = np.reshape(state, [1, *state_size])
done = False
n_steps = 0

while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = preprocess_frame(next_state)
    next_state = np.reshape(next_state, [1, *state_size])
    agent.remember(state, action, reward, next_state, done)
    state = next_state

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    n_steps += 1

print(f"Episode finished after {n_steps} steps")