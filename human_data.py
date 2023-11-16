import gymnasium as gym
import numpy as np
import tkinter as tk

# 创建环境
env = gym.make('PongDeterministic-v4', render_mode='human')

# 初始化状态
state, _ = env.reset()

# 创建tkinter窗口以捕获按键
window = tk.Tk()
window.title("Pong Control")

# 存储当前动作的变量
current_action = 0  # No-op

# 分别用于存储状态、动作、奖励、新状态和是否完成的列表
states = []
actions = []
rewards = []
next_states = []
dones = []

# 根据按键更新动作的函数
def key_pressed(event):
    global current_action
    if event.keysym == 'Up':
        current_action = 2  # Pong中的'向上'动作
    elif event.keysym == 'Down':
        current_action = 3  # Pong中的'向下'动作
    elif event.keysym == 'q':
        window.quit()  # 关闭窗口

# 按键释放时重置动作的函数
def key_released(event):
    global current_action
    if event.keysym in ['Up', 'Down']:
        current_action = 0  # No-op

# 将按键事件绑定到tkinter窗口
window.bind('<KeyPress>', key_pressed)
window.bind('<KeyRelease>', key_released)

# 运行游戏循环的函数
def game_loop():
    global state, current_action

    # 执行动作并获取新状态、奖励、游戏完成状态
    next_state, reward, done, truncated, info = env.step(current_action)

    # 将数据添加到对应的列表
    states.append(state)
    actions.append(current_action)
    rewards.append(reward)
    next_states.append(next_state)
    dones.append(done)

    # 更新当前状态
    state = next_state

    # 渲染游戏屏幕
    env.render()

    if not done:
        window.after(100, game_loop)
    else:
        env.close()

# 启动游戏循环
window.after(100, game_loop)
window.mainloop()

# 保存数据
np.save('states.npy', np.array(states))
np.save('actions.npy', np.array(actions))
np.save('rewards.npy', np.array(rewards))
np.save('next_states.npy', np.array(next_states))
np.save('dones.npy', np.array(dones))
print("人类玩家数据已保存。")
