import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# 创建强化学习策略的函数
def simple_strategy(observation):
    # 这里使用一个非常简单的策略作为示例
    # 您可以根据需要替换为更复杂的策略
    return env.action_space.sample()

# 创建环境
env = gym.make('PongNoFrameskip-v4', render_mode='human')

# 初始化状态
state, info = env.reset()

done = False
n_steps = 0

# 游戏循环
while not done:
    # 应用策略来选择动作
    action = simple_strategy(state)

    # 执行动作并获取新状态、奖励、游戏完成状态以及其他信息
    state, reward, done, truncated, info = env.step(action)

    # 打印观察结果
    print(f"State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")

    # 渲染游戏屏幕
    env.render()

    n_steps += 1

# 关闭环境
env.close()
print(f"Episode finished after {n_steps} steps")
