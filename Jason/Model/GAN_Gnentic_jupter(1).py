#%%
##%%
# Cell 1: 导入库和全局配置
import random
import itertools
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sqlalchemy import create_engine, text

# 全局配置
class Config:
    # 修改为使用 pymysql
    DB_URL = 'mysql+pymysql://root:@localhost:3306/pokemon?charset=utf8mb4'
    STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    TYPE_LIST = ["bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
                 "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                 "rock", "steel", "water"]  # 保持与数据集列名一致
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_RANDOM_TEAMS = 200_000
    BATCH_SIZE = 256
    NUM_EPOCHS = 15

    @classmethod
    def validate(cls, df: pd.DataFrame):
        """数据校验"""
        # 校验主键唯一性
        assert df['pokedex_number'].is_unique, "主键pokedex_number不唯一"

        # 校验类型字段长度
        type_columns = ['type1', 'type2']
        for col in type_columns:
            max_len = df[col].str.len().max()
            assert max_len <= 20, f"列 {col} 存在超过20字符的值（最大长度：{max_len}）"

        # 校验对抗属性列存在
        required_columns = [f'against_{t}' for t in cls.TYPE_LIST]
        missing = set(required_columns) - set(df.columns)
        assert not missing, f"缺失对抗属性列：{missing}"

# 初始化数据库连接（添加连接测试）
try:
    engine = create_engine(Config.DB_URL)
    with engine.connect() as test_conn:
        test_conn.execute(text("SELECT 1"))
    print("数据库连接成功")
except Exception as e:
    print(f"数据库连接失败: {str(e)}")
    raise

# 种子设置
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)
#%%
def verify_dtypes(df):
    # 验证类型特征
    type_cols = [f"type{i}_{t}" for i in (1,2) for t in Config.TYPE_LIST]
    for col in type_cols:
        assert df[col].dtype == np.int8, f"{col} 类型错误：{df[col].dtype}"
        
    # 验证能力向量
    assert all(isinstance(x, np.ndarray) for x in df.abilities_vec), "能力向量类型错误"
    
    # 验证数值列
    numeric_cols = ["height_m", "weight_kg", "base_egg_steps",
                   "capture_rate", "experience_growth", "base_happiness", 
                   "generation", "is_legendary"] + Config.STATS
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} 不是数值类型"
#%%
class DatabaseLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.top_abil = None

    def _execute_query(self, query):
        with self.engine.connect() as conn:
            return conn.execute(text(query))

    def load_data(self):
        query = "SELECT * FROM pokemon"
        self.df = pd.read_sql(query, self.engine)
        self._preprocess_data()
        return self.df, self.top_abil

    def _preprocess_data(self):
        # 基础数据清洗
        self.df["type2"] = self.df["type2"].fillna("None")
        
        # 处理技能列表
        self._process_abilities()
        
        # 类型特征处理
        self._process_types()
        
        # 数值列处理
        self._process_numeric()
        
        # 最终验证
        self._final_validation()

    def _process_abilities(self):
        """处理技能相关特征"""
        # 解析原始技能数据
        def parse_abilities(s):
            if isinstance(s, str):
                try:
                    return json.loads(s.replace("'", '"'))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for string: {s}. Error: {e}")
                    return []
            else:
                return []

        self.df["abilities"] = self.df["abilities"].apply(parse_abilities)

        # 构建top50技能向量
        all_abil = [a for lst in self.df.abilities for a in lst]
        self.top_abil = pd.Series(all_abil).value_counts().nlargest(50).index.tolist()

        # 转换为float32 numpy数组
        self.df["abilities_vec"] = self.df["abilities"].apply(
            lambda lst: np.array(
                [1.0 if ab in lst else 0.0 for ab in self.top_abil],
                dtype=np.float32
            )
        )

    def _process_types(self):
        """处理类型相关特征"""
        # 生成类型特征列
        type_dummies = pd.get_dummies(
            self.df[["type1", "type2"]],
            prefix=["type1", "type2"],
            columns=["type1", "type2"],
            dtype=np.int8
        )

        # 确保包含所有可能类型
        for t in Config.TYPE_LIST:
            for prefix in ["type1", "type2"]:
                col = f"{prefix}_{t}"
                if col not in type_dummies:
                    type_dummies[col] = 0

        # 确保所有列的数据类型为 np.int8
        for col in type_dummies.columns:
            type_dummies[col] = type_dummies[col].astype(np.int8)

        self.df = pd.concat([self.df, type_dummies], axis=1)

    def _process_numeric(self):
        """处理数值特征"""
        numeric_cols = [
            "height_m", "weight_kg", "base_egg_steps",
            "capture_rate", "experience_growth", "base_happiness",
            "generation", "is_legendary"
        ] + Config.STATS
        
        # 强制转换为float32
        self.df[numeric_cols] = self.df[numeric_cols].apply(
            pd.to_numeric, errors='coerce', downcast='float'
        ).fillna(0).astype(np.float32)
        
        # 标准化统计值
        stats_mean = self.df[Config.STATS].mean()
        stats_std = self.df[Config.STATS].std()
        self.df[Config.STATS] = (self.df[Config.STATS] - stats_mean) / stats_std

    def _final_validation(self):
        """最终数据验证"""
        # 验证技能向量
        assert all(isinstance(x, np.ndarray) for x in self.df.abilities_vec), "技能向量必须为numpy数组"
        assert all(x.dtype == np.float32 for x in self.df.abilities_vec), "技能向量必须为float32"
        
        # 验证类型特征
        type_cols = [f"type1_{t}" for t in Config.TYPE_LIST] + [f"type2_{t}" for t in Config.TYPE_LIST]
        assert all(self.df[col].dtype == np.int8 for col in type_cols), "类型特征必须为int8"
        
        # 验证数值列
        numeric_cols = Config.STATS + ["generation", "is_legendary"]
        assert all(pd.api.types.is_float_dtype(self.df[col]) for col in numeric_cols), "数值列必须为float32"
#%%
##%%
# Cell 3: TypeEffectiveness类定义
class TypeEffectiveness:
    def __init__(self):
        self.matrix = self._build_matrix()
    
    def _build_matrix(self):
        eff = np.ones((18, 18))
        eff[0, 9] = 2.0   # Bug vs Grass
        eff[0, 14] = 2.0  # Bug vs Psychic
        eff[0, 1] = 2.0   # Bug vs Dark
        return torch.tensor(eff, dtype=torch.float32)
    
    def get_multiplier(self, attack_type, defend_type):
        try:
            a_idx = Config.TYPE_LIST.index(attack_type)
            d_idx = Config.TYPE_LIST.index(defend_type)
            # 检查索引是否在有效范围内
            if a_idx < 0 or a_idx >= self.matrix.shape[0] or d_idx < 0 or d_idx >= self.matrix.shape[1]:
                print(f"无效的索引：攻击类型 {attack_type} 索引 {a_idx}，防御类型 {defend_type} 索引 {d_idx}")
                return 1.0  # 返回默认值
            return self.matrix[a_idx, d_idx].item()
        except ValueError:
            print(f"未知的类型：攻击类型 {attack_type}，防御类型 {defend_type}")
            return 1.0  # 返回默认值
#%%
##%%
# Cell 4: BattleSimulator类定义
class BattleSimulator:
    def __init__(self, type_matrix):
        self.type_matrix = type_matrix
    
    def duel_prob(self, p_row, q_row):
        is_phys = random.random() < 0.5
        a_stat = "attack" if is_phys else "sp_attack"
        d_stat = "defense" if is_phys else "sp_defense"

        p_types = [p_row["type1"], p_row["type2"]]
        atk_type = random.choice(p_types)
        if atk_type == "None":
            atk_type = random.choice(Config.TYPE_LIST)
            stab = 1.0
        else:
            stab = 1.5

        mult = 1.0
        for q_type in [q_row["type1"], q_row["type2"]]:
            if q_type != "None":
                mult *= self.type_matrix.get_multiplier(atk_type, q_type)

        attack_power = (stab * mult * (p_row[a_stat] + 1))
        defense_power = (q_row[d_stat] + 1)
        damage_ratio = attack_power / defense_power
        return damage_ratio / (damage_ratio + 1)

    def team_vs_team(self, team_F, team_E, df):
        score = 0.0
        for f in team_F:
            for e in team_E:
                score += self.duel_prob(df.loc[f], df.loc[e])
        return score / 36.0
#%%
class PokemonDataset(Dataset):
    def __init__(self, df, type_matrix, top_abil, samples):
        super().__init__()
        self.df = df.reset_index(drop=True)  # 确保索引连续
        self.type_matrix = type_matrix
        self.top_abil = top_abil
        self.samples = samples
        self.simulator = BattleSimulator(type_matrix)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        team_F, team_E = self.samples[idx]
        return self._build_graph(team_F, team_E)

    def _build_graph(self, team_F, team_E):
        # 获取对战双方的宝可梦数据
        team_indices = list(team_F) + list(team_E)
        rows = self.df.iloc[team_indices]  # 使用iloc避免索引问题
        
        # 构建节点特征
        x = torch.stack([self._row_to_tensor(row) for _, row in rows.iterrows()])

        # 构建边连接
        edge_index, edge_attr = self._build_edges(rows)
        
        # 构建图数据
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([self._calculate_battle_score(team_F, team_E)], dtype=torch.float32)
        )
        return data

    def _row_to_tensor(self, row):
        """将单行数据转换为特征张量"""
        # 标量特征
        scalar_features = [
            row["height_m"],
            row["weight_kg"],
            row["base_egg_steps"],
            row["capture_rate"],
            row["experience_growth"],
            row["base_happiness"],
            float(row["generation"]),  # 确保为浮点数
            float(row["is_legendary"]) # 确保为浮点数
        ]
        
        # 统计值特征
        stats_features = [row[stat] for stat in Config.STATS]
        
        # 类型特征
        type_features = [
            row[f"type1_{t}"] for t in Config.TYPE_LIST
        ] + [
            row[f"type2_{t}"] for t in Config.TYPE_LIST
        ]
        
        # 技能向量
        ability_features = row["abilities_vec"].tolist()  # numpy数组转list
        
        # 合并所有特征
        all_features = scalar_features + stats_features + type_features + ability_features
        
        # 转换为张量
        return torch.tensor(all_features, dtype=torch.float32)

    def _build_edges(self, rows):
        """构建全连接边"""
        n = len(rows)
        edge_index = []
        edge_attr = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # 获取攻击方和防御方
                attacker = rows.iloc[i]
                defender = rows.iloc[j]
                
                # 计算类型克制倍数
                type_multiplier = 1.0
                for def_type in [defender["type1"], defender["type2"]]:
                    if def_type != "None":
                        type_multiplier *= self.type_matrix.get_multiplier(
                            attacker["type1"], 
                            def_type
                        )
                
                # 计算属性差
                stat_diff = attacker["attack"] - defender["defense"]
                
                edge_index.append([i, j])
                edge_attr.append([type_multiplier, stat_diff])
        
        return (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            torch.tensor(edge_attr, dtype=torch.float32)
        )

    def _calculate_battle_score(self, team_F, team_E):
        """计算真实对战得分"""
        return self.simulator.team_vs_team(team_F, team_E, self.df)
#%%
##%%
# Cell 6: GATRegression模型定义
class GATRegression(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8):
        super().__init__()
        self.g1 = GATConv(in_channels, hidden, heads=heads, dropout=0.1)
        self.g2 = GATConv(hidden*heads, hidden, heads=1, dropout=0.1)
        self.lin1 = torch.nn.Linear(hidden, 128)
        self.lin2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.g1(x, edge_index))
        x = F.elu(self.g2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)
#%%
##%%
# Cell 7: Trainer类定义
class Trainer:
    def __init__(self, model, dataset):
        self.model = model.to(Config.DEVICE)
        self.loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train(self, epochs=Config.NUM_EPOCHS):
        self.model.train()
        for epoch in range(epochs):
            loss_cum = 0
            for batch in self.loader:
                batch = batch.to(Config.DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(batch)
                loss = F.mse_loss(pred, batch.y)
                loss.backward()
                self.optimizer.step()
                loss_cum += loss.item() * batch.num_graphs
            print(f"Epoch {epoch:02d}  MSE={loss_cum/len(self.loader.dataset):.4f}")
#%%
##%%
# Cell 8: GeneticOptimizer类定义
class GeneticOptimizer:
    def __init__(self, model, df, type_matrix, top_abil):
        self.model = model.to(Config.DEVICE)
        self.df = df
        self.type_matrix = type_matrix
        self.top_abil = top_abil
        self.simulator = BattleSimulator(type_matrix)

    def optimize(self, enemy_team, pop_size=500, generations=150, mutate_p=0.2):
        pool = [p for p in self.df.index if p not in enemy_team]
        population = [tuple(random.sample(pool, 6)) for _ in range(pop_size)]

        for gen in range(generations):
            scored = [(t, self._fitness(t, enemy_team)) for t in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_team, best_fit = scored[0]
            print(f"Gen {gen:03d}  best_fit={best_fit:.4f}")

            elite = [t for t, _ in scored[:pop_size//5]]
            population = self._evolve_population(elite, pool, pop_size, mutate_p)

        return best_team, best_fit

    def _fitness(self, team_F, team_E):
        g = PokemonDataset(self.df, self.type_matrix, self.top_abil, 
                          [(team_F, team_E)]).get(0).to(Config.DEVICE)
        with torch.no_grad():
            return self.model(g).item()

    def _evolve_population(self, elite, pool, pop_size, mutate_p):
        new_pop = elite.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elite, 2)
            child = self._crossover(p1, p2)
            if random.random() < mutate_p:
                child = self._mutate(child, pool)
            new_pop.append(child)
        return new_pop

    def _crossover(self, p1, p2):
        cut = random.randint(1, 5)
        return tuple(list(p1)[:cut] + [x for x in p2 if x not in p1][:6-cut])

    def _mutate(self, child, pool):
        idx = random.randrange(6)
        replacement = random.choice(pool)
        while replacement in child:
            replacement = random.choice(pool)
        return tuple(list(child)[:idx] + [replacement] + list(child)[idx+1:])
#%%
##%%
# Cell 9: 初始化模块并加载数据
db_loader = DatabaseLoader()
df, top_abil = db_loader.load_data()  # 此时会执行新增的类型验证
type_matrix = TypeEffectiveness()
#%%
# Cell 10: 生成训练数据
samples = [
    (tuple(random.sample(df.index.tolist(), 6)),
     tuple(random.sample(df.index.tolist(), 6)))
    for _ in range(Config.NUM_RANDOM_TEAMS)
]
dataset = PokemonDataset(df, type_matrix, top_abil, samples)
#%%
def ultimate_validation(dataset):
    try:
        sample = dataset.get(0)
        print("=== 验证通过 ===")
        return True
    except Exception as e:
        print(f"=== 验证失败：{str(e)} ===")
        # 逐层诊断
        rows = dataset.df.loc[dataset.samples[0][0] + dataset.samples[0][1]]
        for idx, row in rows.iterrows():
            try:
                tensor = dataset._row_to_tensor(row)
                print(f"行 {idx} 转换成功")
            except Exception as ve:
                print(f"行 {idx} 转换失败：{str(ve)}")
                print("问题数据详情：")
                print("能力向量:", type(row["abilities_vec"]), row["abilities_vec"].dtype)
                print("类型特征:", row[[f"type1_{t}" for t in Config.TYPE_LIST[:3]]])
        return False

# 执行验证
ultimate_validation(dataset)
#%%
def validate_tensor_construction(dataset):
    sample_data = dataset.get(0)
    
    print("\n=== 节点特征验证 ===")
    print("特征张量类型:", sample_data.x.dtype)  # 应显示torch.float32
    print("特征形状:", sample_data.x.shape)     # 应为(12, 特征维度)
    
    print("\n=== 边特征验证 ===")
    print("边属性类型:", sample_data.edge_attr.dtype)  # 应显示torch.float32
    
    print("\n=== 类型特征采样验证 ===")
    type_cols = [f"type1_{t}" for t in Config.TYPE_LIST[:3]]
    print(df[type_cols].head(3).values)  # 应显示0/1的整数
    
    print("\n=== 能力向量验证 ===")
    print("向量类型示例:", type(df.abilities_vec.iloc[0]))  # 应显示numpy.ndarray
    print("元素类型示例:", df.abilities_vec.iloc[0].dtype)  # 应显示float32

validate_tensor_construction(dataset)
#%%
##%%
# Cell 11: 初始化并训练模型
model = GATRegression(dataset.get(0).num_node_features)
trainer = Trainer(model, dataset)
trainer.train()
#%%
##%%
# Cell 12: 运行遗传算法优化
enemy_team = df[df.name.isin(["Pikachu","Charizard"])].index.tolist()[:6]
optimizer = GeneticOptimizer(model, df, type_matrix, top_abil)
best_team, score = optimizer.optimize(tuple(enemy_team))
#%%
##%%
# Cell 13: 输出最终结果
print("\n=== 推荐克制队伍 ===")
print(df.loc[list(best_team), "name"].tolist())
print(f"预测胜率: {score:.3f}")