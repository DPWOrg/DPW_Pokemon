import pandas as pd

from VITS_fast_finetune.VITS_fast_finetune.venv.Lib import os

# 加载数据,我直接按我方便的来的
file_path = 'pokemon.csv'
df = pd.read_csv(file_path)
# 计算种族值
df['total_stats'] = df['attack'] + df['defense'] + df['hp'] + df['speed'] + df['sp_attack'] + df['sp_defense']
# 保留战斗相关数值所需要的列
result_df = df[[
    'name',
    'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy',
    'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass',
    'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic',
    'against_rock', 'against_steel', 'against_water',
    'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense', 'total_stats'
]]
selected_df = df[result_df.columns]
# 检查缺失值
missing_values = selected_df.isnull().sum()
total_missing = missing_values.sum()
if total_missing == 0:
    print("没有缺失值")
else:
    print("各列的缺失值数量：")
    print(missing_values)
# 保存战斗数值文件
output_csv = 'battle_skill.csv'
selected_df.to_csv(output_csv, index=False)
# 保存分析宝可梦种族值可能相关的列，用于分析宝可梦种族值与可能有相关性的值
result2_df = df[['name','total_stats', 'is_legendary', 'base_happiness', 'experience_growth', 'base_egg_steps', 'capture_rate', 'weight_kg', 'height_m']]
# 检查缺失值
missing_values = result2_df.isnull().sum()
total_missing = missing_values.sum()
if total_missing == 0:
    print("没有缺失值")
else:
    print("各列的缺失值数量：")
    print(missing_values)
    # 删除有缺失值的行
    result2_df = result2_df.dropna()
# 保存分析宝可梦种族值相关性信息文件
csv_path = 'pokemon_filtered.csv'
result2_df.to_csv(csv_path, index=False)

# 创建文件夹
folder_path = 'pokemon_types'
os.makedirs(folder_path, exist_ok=True)

# 获取 type1 和 type2 的所有唯一值
types = pd.concat([df['type1'], df['type2']]).dropna().unique()

# 遍历每个属性
for pokemon_type in types:
    # 筛选出 type1 或 type2 为当前属性的宝可梦
    type_df = df[(df['type1'] == pokemon_type) | (df['type2'] == pokemon_type)]

    # 提取需要的列
    result_df = type_df[['name', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense', 'total_stats']]

    # 构建文件名
    file_name = f'{pokemon_type}.csv'
    file_path = os.path.join(folder_path, file_name)

    # 保存为 CSV 文件
    result_df.to_csv(file_path, index=False)