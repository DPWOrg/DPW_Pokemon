#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构后的模块化架构：
1. DatabaseLoader - 数据库连接和数据加载
2. TypeEffectiveness - 类型克制矩阵处理
3. PokemonDataset   - 图数据集生成
4. GATRegression     - 图注意力网络模型
5. BattleSimulator   - 战斗模拟逻辑
6. GeneticOptimizer  - 遗传算法优化器
7. Trainer           - 模型训练模块
"""

import random
import itertools
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sqlalchemy import create_engine, text

# 全局配置
class Config:
    DB_URL = 'mysql+mysqlconnector://root:@localhost:3306/pokemon'
    STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    TYPE_LIST = ["bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
                "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                "rock", "steel", "water"]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_RANDOM_TEAMS = 200_000
    BATCH_SIZE = 256
    NUM_EPOCHS = 15

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# 1. 数据库加载模块
class DatabaseLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.top_abil = None

    def _execute_query(self, query):
        with self.engine.connect() as conn:
            return conn.execute(text(query))

    def load_data(self):
        # 从数据库加载原始数据
        query = "SELECT * FROM pokemon"
        self.df = pd.read_sql(query, self.engine)
        
        # 数据预处理
        self._preprocess_data()
        return self.df, self.top_abil

    def _preprocess_data(self):
        # 类型处理
        self.df["type2"] = self.df["type2"].fillna("None")
        
        # 技能处理
        self.df["abilities"] = self.df["abilities"].apply(
            lambda s: json.loads(s.replace("'", '"')) if isinstance(s, str) else []
        )
        
        # 构建技能向量
        all_abil = [a for lst in self.df.abilities for a in lst]
        self.top_abil = pd.Series(all_abil).value_counts().nlargest(50).index.tolist()
        self.df["abilities_vec"] = self.df["abilities"].apply(
            lambda lst: [1 if ab in lst else 0 for ab in self.top_abil]
        )

        # 类型独热编码
        for t in Config.TYPE_LIST:
            self.df[f"type1_{t}"] = (self.df["type1"] == t).astype(int)
            self.df[f"type2_{t}"] = (self.df["type2"] == t).astype(int)

        # 数值处理
        numeric_cols = ["height_m", "weight_kg", "base_egg_steps",
                       "capture_rate", "experience_growth", "base_happiness", 
                       "generation", "is_legendary"] + Config.STATS
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        
        # 标准化
        stats_mean = self.df[Config.STATS].mean()
        stats_std = self.df[Config.STATS].std()
        self.df[Config.STATS] = (self.df[Config.STATS] - stats_mean) / stats_std

# 2. 类型克制矩阵模块
class TypeEffectiveness:
    def __init__(self):
        self.matrix = self._build_matrix()
    
    def _build_matrix(self):
        eff = np.ones((18, 18))
        # 这里填充实际克制关系（示例）
        eff[0, 9] = 2.0   # Bug vs Grass
        eff[0, 14] = 2.0  # Bug vs Psychic
        eff[0, 1] = 2.0   # Bug vs Dark
        return torch.tensor(eff, dtype=torch.float32)
    
    def get_multiplier(self, attack_type, defend_type):
        a_idx = Config.TYPE_LIST.index(attack_type)
        d_idx = Config.TYPE_LIST.index(defend_type)
        return self.matrix[a_idx, d_idx].item()

# 3. 战斗模拟模块
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

# 4. 图数据集模块
class PokemonDataset(Dataset):
    def __init__(self, df, type_matrix, top_abil, samples):
        super().__init__()
        self.df = df
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
        rows = self.df.loc[list(team_F)+list(team_E)]
        x = torch.stack([self._row_to_tensor(r) for _, r in rows.iterrows()])

        n = 12
        send, recv, edge_attr = [], [], []
        for i,j in itertools.product(range(n), range(n)):
            if i==j: continue
            send.append(i); recv.append(j)
            p = rows.iloc[i]; q = rows.iloc[j]
            mult = 1.0
            for q_type in [q["type1"], q["type2"]]:
                if q_type!="None":
                    mult *= self.type_matrix.get_multiplier(p["type1"], q_type)
            dstat = float(p["attack"] - q["defense"])
            edge_attr.append([mult, dstat])
        
        edge_index = torch.tensor([send, recv], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor([self.simulator.team_vs_team(team_F, team_E, self.df)],
                             dtype=torch.float32)
        return data

    def _row_to_tensor(self, row):
        scalar = torch.tensor([
            row["height_m"], row["weight_kg"], row["base_egg_steps"],
            row["capture_rate"], row["experience_growth"],
            row["base_happiness"], row["generation"], row["is_legendary"]
        ], dtype=torch.float32)
        
        stats = torch.tensor(row[Config.STATS].values, dtype=torch.float32)
        types = torch.tensor(row[[f"type{i}_{t}" for i in (1,2) 
                                for t in Config.TYPE_LIST]].values, dtype=torch.float32)
        abil = torch.tensor(row["abilities_vec"], dtype=torch.float32)
        return torch.cat([scalar, stats, types, abil])

# 5. 模型定义
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

# 6. 训练模块
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

# 7. 遗传算法优化器
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

# 主程序
if __name__ == "__main__":
    # 初始化各模块
    db_loader = DatabaseLoader()
    df, top_abil = db_loader.load_data()
    
    type_matrix = TypeEffectiveness()
    
    # 生成训练数据
    samples = [tuple(random.sample(df.index,6)) for _ in range(Config.NUM_RANDOM_TEAMS)]
    dataset = PokemonDataset(df, type_matrix, top_abil, samples)
    
    # 训练模型
    model = GATRegression(dataset.get(0).num_node_features)
    trainer = Trainer(model, dataset)
    trainer.train()
    
    # 遗传算法优化
    enemy_team = df[df.name.isin(["Pikachu","Charizard"])].index.tolist()[:6]
    optimizer = GeneticOptimizer(model, df, type_matrix, top_abil)
    best_team, score = optimizer.optimize(tuple(enemy_team))
    
    print("\n=== 推荐克制队伍 ===")
    print(df.loc[list(best_team), "name"].tolist())
    print(f"预测胜率: {score:.3f}")