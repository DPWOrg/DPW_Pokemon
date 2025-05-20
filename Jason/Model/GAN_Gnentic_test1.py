#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline:
 1. load_and_clean()  –  select columns, encode categorical
 2. build_type_matrix() – construct 18×18 effectiveness matrix
 3. simulate_match()   –  Monte‑Carlo-like 6v6 score for label
 4. generate_dataset() –  random teams → graph → label
 5. GATRegression      –  Graph Attention Net predicting win prob
 6. train()            –  train on generated data
 7. genetic_search()   –  use model to find best counter‑team
"""

import random, itertools, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# 0. GLOBALS
CSV_PATH      = "D:\\python\\Code\\DPW_Pokemon\\archive\\pokemon.csv"
NUM_RANDOM_TEAMS = 200_000        # label generation
SIM_EPOCHS_PER_PAIR = 200         # per P‑Q duel iterations
BATCH_SIZE    = 256
NUM_EPOCHS    = 15
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED          = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

STATS = ["hp","attack","defense","sp_attack","sp_defense","speed"]
TYPE_LIST = ["bug","dark","dragon","electric","fairy","fight","fire","flying",
             "ghost","grass","ground","ice","normal","poison","psychic",
             "rock","steel","water"]


# 1. DATA LOADING + CLEAN 
def load_and_clean(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    
    # 保留核心列 -------------------------------------------------------------
    keep = ["name", "type1", "type2", "abilities", "height_m", "weight_kg",
            "base_egg_steps", "capture_rate", "experience_growth",
            "base_happiness", "generation", "is_legendary"] + STATS
    df = df[keep]

    # 处理类型和技能 ---------------------------------------------------------
    # 类型填充
    df["type2"].fillna("None", inplace=True)
    
    # 技能列表解析
    df["abilities"] = df["abilities"].apply(lambda s: json.loads(s.replace("'", '"')))
    
    # 构建技能向量（Top50）
    all_abil = [a for lst in df.abilities for a in lst]
    top50 = pd.Series(all_abil).value_counts().nlargest(50).index.tolist()
    df["abilities_vec"] = df["abilities"].apply(
        lambda lst: [1 if ab in lst else 0 for ab in top50]
    )

    # 类型独热编码 -----------------------------------------------------------
    for t in TYPE_LIST:
        df[f"type1_{t}"] = (df["type1"] == t).astype(int)
        df[f"type2_{t}"] = (df["type2"] == t).astype(int)

    # 数值清洗处理 ------------------------------------------------------------
    # 捕获率处理（"255 (33.3%)" → 255）
    df["capture_rate"] = df["capture_rate"].str.split().str[0]
    
    # 定义所有数值列（包含STATS）
    numeric_cols = ["height_m", "weight_kg", "base_egg_steps",
                    "capture_rate", "experience_growth",
                    "base_happiness", "generation", "is_legendary"] + STATS
    
    # 强制转换为数值类型（处理"—"等无效值）
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 确保 STATS 列为数值类型
    df[STATS] = df[STATS].apply(pd.to_numeric, errors="coerce")
    
    # 用列均值填充缺失值
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Z-Score标准化（所有数值列已完成清洗）-------------------------------------
    stats_scaler = df[STATS].mean(), df[STATS].std()
    df[STATS] = (df[STATS] - stats_scaler[0]) / stats_scaler[1]

    return df, top50

# 2. TYPE EFFECTIVENESS MATRIX
def build_type_matrix():
    eff = np.ones((18,18))
    # 仅示例列出几行，完整矩阵请依据官方倍率补全
    # 格式: eff[attacker_idx, defender_idx] = multiplier
    bug, grass, psychic, dark = 0, 9, 14, 1
    eff[bug, grass] = 2.0
    eff[bug, dark]  = 2.0
    eff[bug, psychic] = 2.0
    # ... <省略，其余可自行补全> ...
    return torch.tensor(eff, dtype=torch.float32)

TYPE_MAT = build_type_matrix()


# 3. SIMULATOR (简化)
def duel_prob(p_row, q_row):
    """
    Returns win probability of p against q with toy damage model.
    """
    # choose physical vs special
    is_phys = random.random() < 0.5
    a_stat  = "attack" if is_phys else "sp_attack"
    d_stat  = "defense" if is_phys else "sp_defense"

    # STAB
    p_types = [p_row["type1"], p_row["type2"]]
    stab = 1.5

    # choose one attacking type from p_types
    atk_type = random.choice(p_types)
    if atk_type == "None":
        atk_type = random.choice(TYPE_LIST)
        stab = 1.0

    mult = 1.0
    for q_type in [q_row["type1"], q_row["type2"]]:
        if q_type != "None":
            mult *= TYPE_MAT[TYPE_LIST.index(atk_type),
                             TYPE_LIST.index(q_type)].item()

    attack_power  = (stab * mult * (p_row[a_stat] + 1))
    defense_power = (q_row[d_stat] + 1)
    damage_ratio  = attack_power / defense_power
    win_p = damage_ratio / (damage_ratio + 1)  # sigmoid‑like
    return win_p


def team_vs_team(team_F, team_E, df):
    """
    Expected win rate of team_F vs team_E = mean pairwise win prob
    """
    score = 0.0
    for f in team_F:
        for e in team_E:
            score += duel_prob(df.loc[f], df.loc[e])
    return score / 36.0


# 4. GRAPH CONSTRUCTION 
def row_to_tensor(row, top_abil):
    try:
        stats = torch.tensor(row[STATS].values, dtype=torch.float32)
    except Exception as e:
        print("Error in row_to_tensor:", row[STATS].values)
        raise e
    # stats (6) + scalar feats (5) + type one‑hot (36) + ability top50 (50)
    scalar = torch.tensor([
        row["height_m"], row["weight_kg"], row["base_egg_steps"],
        row["capture_rate"], row["experience_growth"],
        row["base_happiness"], row["generation"], row["is_legendary"]
    ], dtype=torch.float32)
    stats  = torch.tensor(row[STATS].values, dtype=torch.float32)
    types  = torch.tensor(row[[f"type{i}_{t}" for i in (1,2)
                               for t in TYPE_LIST]].values, dtype=torch.float32)
    abil   = torch.tensor(row["abilities_vec"], dtype=torch.float32)
    return torch.cat([scalar, stats, types, abil])

def build_graph(team_F, team_E, df, top_abil):
    """
    Nodes = 12; Edges: complete (directed) within & across.
    """
    rows = df.loc[list(team_F)+list(team_E)]
    x = torch.stack([row_to_tensor(r, top_abil) for _, r in rows.iterrows()])

    n = 12
    send, recv, edge_attr = [], [], []
    for i,j in itertools.product(range(n), range(n)):
        if i==j: continue
        send.append(i); recv.append(j)
        p = rows.iloc[i]; q = rows.iloc[j]
        mult = 1.0
        for q_type in [q["type1"], q["type2"]]:
            if q_type!="None":
                mult *= TYPE_MAT[TYPE_LIST.index(p["type1"]),
                                 TYPE_LIST.index(q_type)].item()
        dstat = float(p["attack"] - q["defense"])
        edge_attr.append([mult, dstat])
    edge_index = torch.tensor([send, recv], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor([team_vs_team(team_F, team_E, df)], dtype=torch.float32)
    return data


# 5. DATASET CLASS (corrected)
class TeamDataset(Dataset):
    def __init__(self, df, top_abil, samples):
        super().__init__()  # Initialize parent class
        self.df = df
        self.top_abil = top_abil
        self.samples = samples  # list of (team_F, team_E)

    def len(self): 
        return len(self.samples)

    def get(self, idx):
        team_F, team_E = self.samples[idx]
        return build_graph(team_F, team_E, self.df, self.top_abil)

def generate_random_samples(df, n_samples=NUM_RANDOM_TEAMS):
    index_pool = list(df.index)
    samples=[]
    for _ in tqdm(range(n_samples),desc="Sampling teams"):
        team_E = tuple(random.sample(index_pool, 6))
        # ensure F not overlap E
        remaining = [i for i in index_pool if i not in team_E]
        team_F = tuple(random.sample(remaining, 6))
        samples.append((team_F, team_E))
    return samples


# 6. MODEL: GAT Regression
class GATRegression(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8):
        super().__init__()
        self.g1 = GATConv(in_channels, hidden, heads=heads, dropout=0.1)
        self.g2 = GATConv(hidden*heads, hidden, heads=1, dropout=0.1)
        self.lin1 = torch.nn.Linear(hidden, 128)
        self.lin2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.g1(x, edge_index))
        x = F.elu(self.g2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)


# 7. TRAINING
def train_model(df, top_abil):
    samples = generate_random_samples(df)
    dataset = TeamDataset(df, top_abil, samples)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GATRegression(dataset.get(0).num_node_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        model.train(); loss_cum=0
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            loss_cum += loss.item()*batch.num_graphs
        print(f"Epoch {epoch:02d}  MSE={loss_cum/len(dataset):.4f}")
    return model


# 8. GENETIC SEARCH
def fitness(team_F, team_E, model, df, top_abil):
    g = build_graph(team_F, team_E, df, top_abil).to(DEVICE)
    with torch.no_grad():
        return model(g).item()

def genetic_search(enemy_team, model, df, top_abil,
                   pop_size=500, generations=150, mutate_p=0.2):
    pool = list(df.index)
    pool = [p for p in pool if p not in enemy_team]  # avoid duplicates

    # init population
    population = [tuple(random.sample(pool, 6)) for _ in range(pop_size)]

    for gen in range(generations):
        scored = [(t, fitness(t, enemy_team, model, df, top_abil)) 
                  for t in population]
        scored.sort(key=lambda x:x[1], reverse=True)
        best_team, best_fit = scored[0]
        print(f"Gen {gen:03d}  best_fit={best_fit:.4f}")
        # selection (top‑elite 20%)
        elite = [t for t,_ in scored[:pop_size//5]]
        new_pop = elite.copy()
        # crossover
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elite, 2)
            cut = random.randint(1,5)
            child = tuple(list(p1)[:cut] + 
                          [x for x in p2 if x not in p1][:6-cut])
            # mutation
            if random.random() < mutate_p:
                idx = random.randrange(6)
                replacement = random.choice(pool)
                while replacement in child:
                    replacement = random.choice(pool)
                child = list(child); child[idx]=replacement; child=tuple(child)
            new_pop.append(child)
        population = new_pop
    return best_team, best_fit


# 9. MAIN
if __name__ == "__main__":
    df, top_abil = load_and_clean()
    print("Data loaded:", df.shape)

    # --- step 1: train evaluator ---
    model = train_model(df, top_abil)
    model.eval()

    # --- step 2: user enemy team (example) ---
    user_team_names = ["Pikachu","Charizard","Garchomp",
                       "Blissey","Greninja","Ferrothorn"]
    enemy_team_idx = tuple(df[df.name.isin(user_team_names)].index)
    assert len(enemy_team_idx)==6, "Not all enemy names found!"

    # --- step 3: search counter‑team ---
    best_team_idx, score = genetic_search(enemy_team_idx, model,
                                          df, top_abil)
    best_team_names = df.loc[list(best_team_idx),"name"].tolist()
    print("\n=== Recommended Counter‑Team ===")
    print(best_team_names, f"\nPredicted win‑prob: {score:.3f}")
