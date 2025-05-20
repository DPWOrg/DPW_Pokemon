import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy import create_engine, text
from tqdm import tqdm
from typing import List, Tuple
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import os


# ==================== 全局配置 ====================
class Config:
    DB_URL ='mysql+pymysql://root:@localhost:3306/pokemon?charset=utf8mb4'
    STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    TYPE_LIST = ["bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
                 "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                 "rock", "steel", "water"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_PATH = 'pokemon_team_optimizer.pth'
    POP_SIZE = 500
    GENERATIONS = 100
    MUTATION_RATE = 0.15
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 50
    TRAIN_SAMPLES = 10000


# ==================== 数据加载与预处理 ====================
class PokemonDataLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.feature_columns = []
        self.type_cols = [f'against_{t}' for t in Config.TYPE_LIST]

    def load_and_preprocess(self) -> pd.DataFrame:
        """适配实际数据结构的预处理"""
        query = "SELECT * FROM pokemon"
        self.df = pd.read_sql(query, self.engine)

        # 重命名列以保持兼容性（实际数据中的列名是type1/type2）
        self.df = self.df.rename(columns={
            'type1': 'original_type1',
            'type2': 'original_type2'
        })

        # 填充空值并标准化类型列
        self.df['original_type2'] = self.df['original_type2'].fillna('None')

        # 特征工程
        self._process_numeric_features()
        self._process_type_features()

        # 构建特征矩阵（分离特征数据和原始数据）
        self._build_feature_matrix()
        return self.df

    def _process_type_features(self):
        """类型特征处理优化"""
        # 使用实际存在的类型列
        type_df = self.df[['original_type1', 'original_type2']].copy()

        # 过滤非法类型值
        valid_types = set(Config.TYPE_LIST + ['None'])
        type_df['original_type1'] = type_df['original_type1'].where(
            type_df['original_type1'].isin(valid_types), 'None')
        type_df['original_type2'] = type_df['original_type2'].where(
            type_df['original_type2'].isin(valid_types), 'None')

        # 生成虚拟变量
        type_dummies = pd.get_dummies(
            type_df,
            prefix=['type1', 'type2'],
            columns=['original_type1', 'original_type2'],
            dtype=np.float32
        )

        # 合并到主数据框（保留原始类型列）
        self.df = pd.concat([self.df, type_dummies], axis=1)

    def _build_feature_matrix(self):
        """构建适配实际数据的特征矩阵"""
        # 定义特征列（根据实际数据调整，移除技能相关列）
        self.feature_columns = (
                Config.STATS +
                [f'type1_{t}' for t in Config.TYPE_LIST] +
                [f'type2_{t}' for t in Config.TYPE_LIST] +
                [f'against_{t}' for t in Config.TYPE_LIST]
        )

        # 确保所有特征列存在
        for col in self.feature_columns:
            if col not in self.df:
                self.df[col] = 0.0

        # 创建纯数值特征矩阵
        self.feature_df = self.df[self.feature_columns].astype(np.float32)
        # 保留原始数据列
        self.df = self.df[['original_type1', 'original_type2'] + self.feature_columns]

    def _process_numeric_features(self):
        """处理数值特征"""
        stats = Config.STATS
        # 强制将这些列转换为数值类型，并处理可能的错误值
        self.df[stats] = self.df[stats].apply(pd.to_numeric, errors='coerce')
        # 过滤掉包含 NaN 的行
        self.df = self.df.dropna(subset=stats)
        self.df[stats] = (self.df[stats] - self.df[stats].mean()) / self.df[stats].std()
        self.df[stats] = self.df[stats].astype(np.float32)


# ==================== 类型计算模块 ====================
class TypeCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_cols = [f'against_{t}' for t in Config.TYPE_LIST]
        self._build_type_matrix()

    def _build_type_matrix(self):
        """构建类型对抗矩阵"""
        # 确保数值类型正确
        self.type_matrix = self.df[self.type_cols].values.astype(np.float32)

    def get_effectiveness(self, attacker_idx: int, defender_types: List[str]) -> float:
        """获取攻击方对防御方类型的总倍率"""
        effectiveness = 1.0
        for t in defender_types:
            if t == 'None':
                continue
            col_idx = Config.TYPE_LIST.index(t)
            effectiveness *= self.type_matrix[attacker_idx, col_idx]
        return effectiveness


# ==================== 战斗模拟器 ====================
class BattleSimulator:
    def __init__(self, type_calculator: TypeCalculator):
        self.tc = type_calculator

    def evaluate_matchup(self, attacker_idx: int, defender_idx: int) -> float:
        """优化后的对战评估函数"""
        # 获取战斗数据（使用原始类型列）
        attacker = self.tc.df.loc[attacker_idx]
        defender = self.tc.df.loc[defender_idx]

        # 物理/特殊攻击判断（使用标准化后的属性值）
        attack_stat = attacker['attack']
        sp_attack_stat = attacker['sp_attack']
        is_physical = attack_stat > sp_attack_stat

        # 获取攻击/防御数值（添加平滑系数防止除零）
        attack_value = attack_stat if is_physical else sp_attack_stat
        defense_value = defender['defense'] if is_physical else defender['sp_defense']
        defense_value = max(defense_value, 0.1)  # 确保最小防御值

        # 获取防御方有效类型（带类型有效性验证）
        defender_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = defender.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                defender_types.append(t)

        # 获取攻击方有效类型（用于STAB计算）
        attacker_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = attacker.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                attacker_types.append(t)

        # 计算类型倍率（带异常处理）
        try:
            type_mult = self.tc.get_effectiveness(attacker_idx, defender_types)
        except IndexError:
            type_mult = 1.0  # 异常时使用中性倍率

        # STAB加成逻辑优化
        if attacker_types:
            move_type = random.choice(attacker_types)
            stab = 1.5 if move_type in attacker_types else 1.0
        else:
            move_type = random.choice(Config.TYPE_LIST)
            stab = 1.0

        # 伤害计算公式优化 (更健壮的处理)
        safe_attack = np.nan_to_num(attack_value, nan=0.0)  # 处理NaN值
        safe_attack = max(abs(safe_attack), 1e-5)  # 确保正值且不小于最小阈值
        base_damage = 0.5 * (safe_attack ** 1.3)  # 攻击力非线性缩放
        defense_factor = (defense_value ** 0.8) + 1e-5  # 防御力次线性影响
        raw_damage = base_damage * stab * type_mult / defense_factor

        # 标准化输出（sigmoid压缩到0-1范围）
        normalized_damage = 1 / (1 + np.exp(-raw_damage / 50 + 3))
        return np.clip(normalized_damage, 0.0, 1.0)

    def team_score(self, team: List[int], enemy_team: List[int]) -> float:
        """团队得分计算"""
        total = 0.0
        for a in team:
            for e in enemy_team:
                total += self.evaluate_matchup(a, e)
        return total / (len(team) * len(enemy_team))


# ==================== 神经网络模型 ====================
class TeamEvaluator(nn.Module):
    """团队评估模型"""

    def __init__(self, input_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim, 128, heads=4)
        self.gat2 = GATConv(128 * 4, 64)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()


class PokemonDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]


class ModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # 根据更新后的特征列数量调整输入维度
        input_dim = len(self.df.columns) - 2  # 减去 'original_type1' 和 'original_type2'
        self.model = TeamEvaluator(input_dim=input_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def generate_training_data(self):
        """生成训练数据"""
        samples = []
        for _ in range(Config.TRAIN_SAMPLES):
            team = np.random.choice(self.df.index, size=6, replace=False)
            enemy = np.random.choice(self.df.index, size=6, replace=False)

            # 创建图数据
            node_features = torch.tensor(
                self.df.loc[np.concatenate([team, enemy])].drop(['original_type1', 'original_type2'], axis=1).values,
                dtype=torch.float32
            )
            edge_index = torch.combinations(torch.arange(node_features.size(0)), 2).t()
            data = Data(
                x=node_features,
                edge_index=edge_index,
                y=torch.tensor([self._calc_real_score(team, enemy)], dtype=torch.float32
                               ))
            samples.append(data)
        return samples

    def _calc_real_score(self, team, enemy):
        """计算真实对战得分"""
        simulator = BattleSimulator(TypeCalculator(self.df))
        return simulator.team_score(team, enemy)

    def train(self):
        """训练模型"""
        dataset = self.generate_training_data()
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(Config.TRAIN_EPOCHS):
            total_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch.to(Config.DEVICE))
                loss = F.mse_loss(outputs, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{Config.TRAIN_EPOCHS} Loss: {total_loss / len(loader):.4f}")

        torch.save(self.model.state_dict(), Config.MODEL_PATH)
        print(f"模型已保存至 {Config.MODEL_PATH}")


# ==================== 遗传算法优化器 ====================
class GeneticOptimizer:
    def __init__(self, df: pd.DataFrame, enemy_team: List[int]):
        self.df = df
        self.enemy_team = enemy_team
        self.tc = TypeCalculator(df)
        self.simulator = BattleSimulator(self.tc)
        self.model = self._try_load_model()

        self.pool = [i for i in df.index if i not in enemy_team]
        self.enemy_types = self._analyze_enemy_types()

    def _try_load_model(self):
        """尝试加载模型"""
        if os.path.exists(Config.MODEL_PATH):
            input_dim = len(self.df.columns) - 2  # 减去 'original_type1' 和 'original_type2'
            model = TeamEvaluator(input_dim=input_dim)
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            return model.eval()
        return None

    def _analyze_enemy_types(self) -> np.ndarray:
        """分析敌方类型分布"""
        type_counts = np.zeros(len(Config.TYPE_LIST))
        for idx in self.enemy_team:
            types = [
                t for t in [
                    self.df.loc[idx]['original_type1'],
                    self.df.loc[idx]['original_type2']
                ] if t != 'None'
            ]
            for t in types:
                try:
                    type_counts[Config.TYPE_LIST.index(t)] += 1
                except ValueError:
                    continue
        return type_counts / max(1, len(self.enemy_team))

    def _initialize_population(self) -> List[List[int]]:
        """智能初始化种群"""
        population = []
        pool_array = np.array(self.pool)

        for _ in range(Config.POP_SIZE):
            # 基于类型相性选择
            scores = self.tc.type_matrix[self.pool] @ self.enemy_types
            if scores.sum() == 0:
                # 如果总和为零，使用均匀分布
                probs = np.ones(len(pool_array)) / len(pool_array)
            else:
                probs = scores / scores.sum()
            selected = np.random.choice(pool_array, 6, p=probs, replace=False)
            population.append(list(selected))
        return population

    def _fitness(self, team: List[int]) -> float:
        """适应度函数"""
        if self.model:
            return self._model_predict(team)
        return self.simulator.team_score(team, self.enemy_team)

    def _model_predict(self, team: List[int]) -> float:
        """使用神经网络预测"""
        data = self._create_graph_data(team)
        with torch.no_grad():
            return self.model(data.to(Config.DEVICE)).item()

    def _create_graph_data(self, team: List[int]) -> Data:
        """创建图数据"""
        node_features = torch.tensor(
            self.df.loc[team + self.enemy_team].drop(['original_type1', 'original_type2'], axis=1).values,
            dtype=torch.float32
        )
        edge_index = torch.combinations(torch.arange(node_features.size(0)), 2).t()
        return Data(x=node_features, edge_index=edge_index)

    def optimize(self) -> Tuple[List[int], float]:
        """优化主循环"""
        population = self._initialize_population()
        best_team, best_score = None, -np.inf

        with tqdm(total=Config.GENERATIONS, desc="Optimizing") as pbar:
            for _ in range(Config.GENERATIONS):
                # 评估适应度
                scores = [self._fitness(t) for t in population]

                # 更新最佳团队
                current_best_idx = np.argmax(scores)
                if scores[current_best_idx] > best_score:
                    best_score = scores[current_best_idx]
                    best_team = population[current_best_idx]

                # 选择（锦标赛选择）
                selected = []
                for _ in range(Config.POP_SIZE):
                    candidates = random.sample(population, 3)
                    winner = max(candidates, key=lambda t: self._fitness(t))
                    selected.append(winner)

                # 交叉与变异
                new_pop = []
                for i in range(0, Config.POP_SIZE, 2):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child1, child2 = self._crossover(parent1, parent2)
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    new_pop.extend([child1, child2])

                population = new_pop
                pbar.update(1)

        return best_team, best_score

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """交叉操作"""
        crossover_point = random.randint(1, 5)
        child1 = parent1[:crossover_point] + [p for p in parent2 if p not in parent1[:crossover_point]][:6 - crossover_point]
        child2 = parent2[:crossover_point] + [p for p in parent1 if p not in parent2[:crossover_point]][:6 - crossover_point]
        return child1, child2

    def _mutate(self, team: List[int]) -> List[int]:
        """变异操作"""
        for i in range(len(team)):
            if random.random() < Config.MUTATION_RATE:
                new_pokemon = random.choice(self.pool)
                while new_pokemon in team:
                    new_pokemon = random.choice(self.pool)
                team[i] = new_pokemon
        return team


# 主函数
def main():
    # 加载和预处理数据
    loader = PokemonDataLoader()
    df = loader.load_and_preprocess()

    # 训练模型
    trainer = ModelTrainer(df)
    trainer.train()

    # 示例敌方团队
    enemy_team = [1, 2, 3, 4, 5, 6]

    # 优化团队
    optimizer = GeneticOptimizer(df, enemy_team)
    best_team, best_score = optimizer.optimize()

    print(f"最佳团队: {best_team}")
    print(f"最佳得分: {best_score}")


if __name__ == "__main__":
    main()
