import pandas as pd
import numpy as np
import os

file_path = r"C:\xampp\htdocs\python\battle_skill.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    print(f"文件不存在")
    import sys
    sys.exit(1)

states = df[['against_bug', 'against_dark', 'against_dragon', 'against_electric',
             'against_fairy', 'against_fight', 'against_fire', 'against_flying',
             'against_ghost', 'against_grass', 'against_ground', 'against_ice',
             'against_normal', 'against_poison', 'against_psychic', 'against_rock',
             'against_steel', 'against_water', 'hp', 'attack', 'defense','speed',
             'sp_attack','sp_defense', 'total_stats']].values

# 动作
actions = df['name'].values

# Q表初始化
num_states = states.shape[0]
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))

# 超参数设置
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 1000  # 训练轮数

# 奖励函数
def calculate_reward(action_index, target_pokemon_type_index, states):
    # 针对目标属性的伤害系数
    against_value = states[action_index, target_pokemon_type_index]
    # 总能力值
    total_stats_value = states[action_index, -1]
    # 速度属性
    speed_value = states[action_index, 22]
    # 生命值
    hp_value = states[action_index, 18]
    # 防御属性
    defense_value = states[action_index, 20]

    # 综合考虑各项属性计算奖励
    reward = against_value * 0.5 + total_stats_value * 0.2 / 1000 + speed_value * 0.1 / 100 + hp_value * 0.1 / 100 + defense_value * 0.1 / 100
    return reward

# 训练过程
for episode in range(num_episodes):
    # 随机选择初始状态
    state_index = np.random.randint(0, num_states)
    # 随机选择目标宝可梦及其属性类型
    target_pokemon_index = np.random.randint(0, num_states)
    target_pokemon_type_index = np.random.randint(0, 18)

    for _ in range(10):  # 每个回合的步数
        if np.random.uniform(0, 1) < epsilon:
            # 探索：随机选择动作
            action_index = np.random.randint(0, num_actions)
        else:
            # 利用：选择 Q 值最大的动作
            action_index = np.argmax(Q[state_index, :])

        # 计算奖励
        reward = calculate_reward(action_index, target_pokemon_type_index, states)

        # 随机选择下一个状态
        next_state_index = np.random.randint(0, num_states)

        # 更新 Q 表
        Q[state_index, action_index] = (1 - alpha) * Q[state_index, action_index] + \
                                       alpha * (reward + gamma * np.max(Q[next_state_index, :]))

        state_index = next_state_index

# 获取用户输入的宝可梦名字
input_pokemon_name = input("请输入一个宝可梦的名字: ")

# 查找输入宝可梦的索引
try:
    target_pokemon_index = df[df['name'] == input_pokemon_name].index[0]
    # 随机选择目标宝可梦的属性类型（这里简单假设随机选一个）
    target_pokemon_type_index = np.random.randint(0, 18)
    # 选择 Q 值最大的宝可梦作为针对宝可梦
    best_pokemon_index = np.argmax(Q[:, target_pokemon_type_index])
    best_pokemon_name = actions[best_pokemon_index]
    print(f"目标宝可梦是 {input_pokemon_name}，针对它的最佳宝可梦是 {best_pokemon_name}")
except IndexError:
    print(f"未找到名为 {input_pokemon_name} 的宝可梦，请检查输入是否正确。")
    