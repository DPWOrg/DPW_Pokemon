import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy import create_engine, text
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import os
from collections import defaultdict



# ==================== 全局配置 ====================
class Config:
    DB_URL = 'mysql+pymysql://root:@localhost:3306/pokemon?charset=utf8mb4'
    STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    TYPE_LIST = ["bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
                 "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                 "rock", "steel", "water"]  # 保持与数据库列名一致
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    MODEL_PATH = 'pokemon_team_optimizer.pth'
    POP_SIZE = 500
    GENERATIONS = 100
    MUTATION_RATE = 0.15
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 50
    TRAIN_SAMPLES = 10000
    LEARNING_RATE = 1e-4
    GRAD_CLIP = 1.0
    LOSS_FUNCTION = nn.L1Loss()
    NUM_TEAMS = 5  # 要生成的队伍数量
    WIN_RATE_SIMULATIONS = 100  # 胜率模拟次数


# ==================== 数据加载与预处理 ====================
class PokemonDataLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.feature_columns = []
        self.type_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]

    def load_and_preprocess(self) -> pd.DataFrame:
        """适配实际数据结构的预处理"""
        query = "SELECT * FROM pokemon"
        try:
            self.df = pd.read_sql(query, self.engine).set_index('pokedex_number')
            print(f"成功加载数据，总记录数: {len(self.df)}")
            print("前3条记录样本:\n", self.df.iloc[:3][['hp', 'attack']])  # 移除不存在的列
            print("特征列示例:", self.feature_columns[:5])
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

        # 列名智能匹配
        print("检测到的列:", list(self.df.columns))
        
        # 查找类型列（不区分大小写和符号）
        type1_col = next((col for col in self.df.columns if 'type' in col.lower() and ('1' in col or '一' in col)), None)
        type2_col = next((col for col in self.df.columns if 'type' in col.lower() and ('2' in col or '二' in col)), None)

        rename_map = {}
        if type1_col:
            rename_map[type1_col] = 'original_type1'
        if type2_col:
            rename_map[type2_col] = 'original_type2'

        if rename_map:
            try:
                self.df = self.df.rename(columns=rename_map)
                print(f"成功映射类型列: {rename_map}")
            except Exception as e:
                print(f"列重命名错误: {str(e)}")
                print("使用后备列名...")
                self.df['original_type1'] = self.df.get(type1_col, 'None')
                self.df['original_type2'] = self.df.get(type2_col, 'None')
        else:
            print("警告: 未检测到类型列，创建默认列")
            self.df['original_type1'] = 'None'
            self.df['original_type2'] = 'None'

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
        # 根据实际数据库列名调整对抗属性列
        actual_against_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]
        self.feature_columns = (
                Config.STATS +
                [f'type1_{t}' for t in Config.TYPE_LIST] +
                [f'type2_{t}' for t in Config.TYPE_LIST] +
                actual_against_cols
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
        self.type_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]
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
        attacker = self.tc.df.iloc[attacker_idx]
        defender = self.tc.df.iloc[defender_idx]

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
        self.gat1 = GATConv(input_dim, 64, heads=4)  # 简化模型
        self.gat2 = GATConv(64 * 4, 32)  # 简化模型
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),  # 更换激活函数
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.gat1(x, edge_index))  # 更换激活函数
        x = F.leaky_relu(self.gat2(x, edge_index))  # 更换激活函数
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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.loss_function = Config.LOSS_FUNCTION

    def generate_training_data(self):
        """生成训练数据"""
        samples = []
        for _ in range(Config.TRAIN_SAMPLES):
            team = np.random.choice(self.df.index, size=6, replace=False)
            enemy = np.random.choice(self.df.index, size=6, replace=False)

            # 合并并去重
            combined = list(set(np.concatenate([team, enemy])))
            if len(combined) < 2:  # 至少需要2个节点才能生成边
                continue

            # 创建图数据
            node_features = torch.tensor(
                self.df.iloc[combined].drop(['original_type1', 'original_type2'], axis=1).values,
                dtype=torch.float32
            )
            edge_index = torch.combinations(torch.arange(len(combined)), 2).t()
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
        print("开始生成训练数据...")
        try:
            dataset = self.generate_training_data()
            if not dataset:
                raise ValueError("无法生成训练数据，请检查数据加载和预处理")
            print(f"成功生成 {len(dataset)} 条训练数据")
        except Exception as e:
            print(f"训练数据生成失败: {str(e)}")
            return

        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(Config.TRAIN_EPOCHS):
            total_loss = 0
            print(f"\nEpoch {epoch+1}/{Config.TRAIN_EPOCHS}")
            for batch_idx, batch in enumerate(tqdm(loader, desc="训练中")):
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch.to(Config.DEVICE))
                    loss = self.loss_function(outputs, batch.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                    self.optimizer.step()
                    total_loss += loss.item()
                    
                    # 每10个batch输出一次进度
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"训练过程中出现异常（Batch {batch_idx}）: {str(e)}")
                    continue

            print(f"Epoch {epoch + 1}/{Config.TRAIN_EPOCHS} Loss: {total_loss / len(loader):.4f}")

        torch.save(self.model.state_dict(), Config.MODEL_PATH)
        print(f"模型已保存至 {Config.MODEL_PATH}")






# ==================== 新增胜率分析模块 ====================
class WinRateAnalyzer:
    def __init__(self, simulator: 'BattleSimulator'):
        self.simulator = simulator

    def calculate_win_rate(self, team: List[int], enemy_team: List[int],
                           num_simulations=Config.WIN_RATE_SIMULATIONS) -> float:
        """计算指定次数的模拟胜率"""
        wins = 0
        for _ in range(num_simulations):
            team_score = 0
            enemy_score = 0

            # 混战模拟：随机选择攻击方和防御方
            for _ in range(10):  # 每次模拟10次随机对战
                attacker = random.choice(team + enemy_team)
                defender = random.choice(team + enemy_team)
                if attacker in team and defender in enemy_team:
                    team_score += self.simulator.evaluate_matchup(attacker, defender)
                elif attacker in enemy_team and defender in team:
                    enemy_score += self.simulator.evaluate_matchup(attacker, defender)

            if team_score > enemy_score:
                wins += 1

        return wins / num_simulations

    def batch_calculate_win_rates(self, teams: List[List[int]], enemy_team: List[int]) -> List[float]:
        """批量计算胜率"""
        return [self.calculate_win_rate(team, enemy_team) for team in teams]


# ==================== 队伍分析模块 ====================
class TeamAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_cols = [f'type1_{t}' for t in Config.TYPE_LIST] + [f'type2_{t}' for t in Config.TYPE_LIST]

    def get_team_analysis(self, team: List[int]) -> Dict:
        """获取队伍的详细分析"""
        members = self.df.loc[team]
        return {
            "type_distribution": self._get_type_distribution(members),
            "stat_summary": self._get_stat_summary(members),
            "type_coverage": self._get_type_coverage(members)
        }

    def _get_type_distribution(self, members: pd.DataFrame) -> Dict[str, float]:
        """获取类型分布"""
        type_counts = defaultdict(int)
        for _, row in members.iterrows():
            if row['original_type1'] != 'None':
                type_counts[row['original_type1']] += 1
            if row['original_type2'] != 'None':
                type_counts[row['original_type2']] += 1
        total = sum(type_counts.values())
        return {t: count / total for t, count in type_counts.items()}

    def _get_stat_summary(self, members: pd.DataFrame) -> Dict[str, float]:
        """获取属性统计"""
        stats = Config.STATS
        return {
            stat: {
                'mean': members[stat].mean(),
                'max': members[stat].max(),
                'min': members[stat].min()
            } for stat in stats
        }

    def _get_type_coverage(self, members: pd.DataFrame) -> Dict[str, float]:
        """获取类型覆盖情况"""
        coverage = defaultdict(float)
        for _, row in members.iterrows():
            for t in Config.TYPE_LIST:
                coverage[t] = max(coverage[t], row[f'against_{t}'])
        return coverage

    def generate_advice(self, team: List[int], enemy_team: List[int]) -> str:
        """生成队伍建议"""
        team_analysis = self.get_team_analysis(team)
        enemy_analysis = self.get_team_analysis(enemy_team)

        advice = []
        # 类型建议
        team_types = set(team_analysis['type_distribution'].keys())
        enemy_types = set(enemy_analysis['type_distribution'].keys())
        missing_coverage = enemy_types - team_types
        if missing_coverage:
            advice.append(f"建议补充克制 {', '.join(missing_coverage)} 类型的宝可梦")

        # 属性建议
        defense_weakness = team_analysis['stat_summary']['defense']['min']
        if defense_weakness < -1.0:
            advice.append("队伍物理防御较弱，建议增加防御型宝可梦")

        sp_defense_weakness = team_analysis['stat_summary']['sp_defense']['min']
        if sp_defense_weakness < -1.0:
            advice.append("队伍特殊防御较弱，建议增加特防型宝可梦")

        return "\n".join(advice) if advice else "队伍属性较为均衡"


# ==================== 遗传算法优化器 ====================
class GeneticOptimizer:
    def _try_load_model(self):
        """尝试加载模型"""
        if os.path.exists(Config.MODEL_PATH):
            input_dim = len(self.df.columns) - 2  # 减去 'original_type1' 和 'original_type2'
            model = TeamEvaluator(input_dim=input_dim)
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            return model.eval()
        return None

    def __init__(self, df: pd.DataFrame, enemy_team: List[int]):
        self.df = df
        self.enemy_team = enemy_team
        self.tc = TypeCalculator(df)
        self.simulator = BattleSimulator(self.tc)
        self.win_analyzer = WinRateAnalyzer(self.simulator)
        self.team_analyzer = TeamAnalyzer(df)
        self.model = self._try_load_model()
        self.pool = [i for i in df.index if i not in enemy_team]
        self.enemy_types = self._analyze_enemy_types()
        self.unique_teams = set()  # 用于记录唯一队伍

    def _analyze_enemy_types(self) -> np.ndarray:
        """分析敌方类型分布"""
        from __main__ import Config  # 添加Config类的引用
        type_counts = np.zeros(len(Config.TYPE_LIST))
        for idx in self.enemy_team:
            types = [
                t for t in [
                    self.df.iloc[idx]['original_type1'],
                    self.df.iloc[idx]['original_type2']
                ] if t != 'None'
            ]
            for t in types:
                try:
                    type_counts[Config.TYPE_LIST.index(t)] += 1
                except ValueError:
                    continue
        return type_counts / max(1, len(self.enemy_team))

    def _initialize_population(self) -> List[List[int]]:
        """初始化种群：生成随机宝可梦队伍"""
        population = []
        for _ in range(Config.POP_SIZE):
            # 确保队伍不重复且不包含敌方成员
            team = np.random.choice(self.pool, size=6, replace=False).tolist()
            while frozenset(team) in self.unique_teams:
                team = np.random.choice(self.pool, size=6, replace=False).tolist()
            population.append(team)
        return population

    def _fitness(self, team: List[int]) -> float:
        """计算队伍适应度：结合类型优势和模型预测"""
        from __main__ import Config  # 添加Config类的引用
        
        # 类型得分：针对敌方类型的克制能力
        type_score = 0.0
        team_types = self._get_team_types(team)
        for t in team_types:
            if t in Config.TYPE_LIST:
                # Get actual type name from index
                enemy_type = Config.TYPE_LIST[self.enemy_types.argmax()]
                type_score += self.tc.get_effectiveness(
                    Config.TYPE_LIST.index(t),
                    [enemy_type]  # Pass as list of type strings
                ) if t in Config.TYPE_LIST else 0.0
        
        # 属性得分：队伍属性平衡性
        stats = self.df.loc[team][Config.STATS].values
        stat_score = np.mean(stats) - 0.5 * np.std(stats)  # 鼓励高均值低方差
        
        # 模型预测得分（如果模型存在）
        model_score = 0.0
        if self.model:
            with torch.no_grad():
                # 动态生成节点索引
                team_features = self.df.loc[team].drop(['original_type1', 'original_type2'], axis=1)
                num_nodes = len(team_features)
                if num_nodes < 2:  # 处理特殊情况
                    return 0.0
                
                # 生成有效边索引
                edge_index = torch.combinations(torch.arange(num_nodes), 2).t()
                data = Data(
                    x=torch.tensor(team_features.values, dtype=torch.float32),
                    edge_index=edge_index
                )
                model_score = self.model(data.to(Config.DEVICE)).item()
        
        # 综合得分权重
        return 0.4 * type_score + 0.3 * stat_score + 0.3 * model_score

    def optimize(self) -> List[Tuple[List[int], float]]:
        """优化主循环，返回多个优秀队伍"""
        population = self._initialize_population()
        best_teams = []

        with tqdm(total=Config.GENERATIONS, desc="Optimizing") as pbar:
            for generation in range(Config.GENERATIONS):
                # 评估适应度
                scores = [self._fitness(t) for t in population]

                # 更新最佳团队（保留前10%）
                sorted_indices = np.argsort(scores)[::-1]
                for idx in sorted_indices[:Config.POP_SIZE // 10]:
                    team = population[idx]
                    team_hash = frozenset(team)
                    if team_hash not in self.unique_teams:
                        self.unique_teams.add(team_hash)
                        best_teams.append((team, scores[idx]))

                # 选择（锦标赛选择）
                selected = self._tournament_selection(population, scores)

                # 交叉与变异
                population = self._crossover_and_mutate(selected, generation)
                pbar.update(1)

        # 后处理：筛选并计算真实胜率
        return self._post_process(best_teams)

    def _post_process(self, candidates: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
        """后处理：去重并计算真实胜率"""
        # 去重处理
        unique = {}
        for team, score in candidates:
            key = frozenset(team)
            if key not in unique or score > unique[key][1]:
                unique[key] = (team, score)

        # 取前N个候选
        sorted_teams = sorted(unique.values(), key=lambda x: x[1], reverse=True)[:Config.NUM_TEAMS * 2]

        # 计算真实胜率
        final_teams = []
        for team, _ in sorted_teams:
            win_rate = self.win_analyzer.calculate_win_rate(team, self.enemy_team)
            final_teams.append((team, win_rate))

        # 按胜率排序并返回前N个
        return sorted(final_teams, key=lambda x: x[1], reverse=True)[:Config.NUM_TEAMS]

    def _tournament_selection(self, population, scores):
        """锦标赛选择"""
        selected = []
        for _ in range(Config.POP_SIZE):
            candidates = random.sample(list(zip(population, scores)), 3)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def _crossover_and_mutate(self, selected, generation):
        """改进的交叉变异操作"""
        new_pop = []
        
        # 多点交叉
        for i in range(0, len(selected), 2):
            if i + 1 >= len(selected):
                break
            parent1, parent2 = selected[i], selected[i + 1]
            
            # 生成两个交叉点
            cross_points = sorted(random.sample(range(1, 6), 2))
            child1 = parent1[:cross_points[0]] + parent2[cross_points[0]:cross_points[1]] + parent1[cross_points[1]:]
            child2 = parent2[:cross_points[0]] + parent1[cross_points[0]:cross_points[1]] + parent2[cross_points[1]:]
            
            # 去除重复ID并补足
            child1 = list(dict.fromkeys(child1))[:6]
            child2 = list(dict.fromkeys(child2))[:6]
            
            # 自适应变异
            new_pop.extend([self._mutate(child1, generation), self._mutate(child2, generation)])
            
        return new_pop

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """改进的交叉操作，保持多样性"""
        common = set(parent1) & set(parent2)
        unique1 = [p for p in parent1 if p not in common]
        unique2 = [p for p in parent2 if p not in common]

        # 保留共有元素，随机补充独特元素
        child1 = list(common) + random.sample(unique1 + unique2, 6 - len(common))
        child2 = list(common) + random.sample(unique1 + unique2, 6 - len(common))

        # 确保不重复
        child1 = list(dict.fromkeys(child1))[:6]
        child2 = list(dict.fromkeys(child2))[:6]
        return child1, child2

    def _mutate(self, team: List[int], generation: int) -> List[int]:
        """改进的变异操作，考虑类型平衡"""
        for i in range(len(team)):
            if random.random() < Config.MUTATION_RATE:
                # 基于类型互补选择新成员
                current_types = self._get_team_types(team)
                candidate_pool = [p for p in self.pool if self._is_type_complementary(p, current_types)]
                if candidate_pool:
                    team[i] = random.choice(candidate_pool)
        return team

    def _get_team_types(self, team: List[int]) -> List[str]:
        """获取队伍所有类型"""
        types = []
        for idx in team:
            types.append(self.df.loc[idx, 'original_type1'])
            if self.df.loc[idx, 'original_type2'] != 'None':
                types.append(self.df.loc[idx, 'original_type2'])
        return list(set(types))

    def _is_type_complementary(self, candidate: int, current_types: List[str]) -> bool:
        """判断候选是否补充队伍类型"""
        candidate_types = [
            self.df.loc[candidate, 'original_type1'],
            self.df.loc[candidate, 'original_type2']
        ]
        candidate_types = [t for t in candidate_types if t != 'None']
        return any(t not in current_types for t in candidate_types)


# ==================== 主函数改进 ====================
def main():
    # 加载和预处理数据
    loader = PokemonDataLoader()
    df = loader.load_and_preprocess()

    # 训练模型（如果不存在）
    if not os.path.exists(Config.MODEL_PATH):
        trainer = ModelTrainer(df)
        trainer.train()

    # 从库里随机抽六个作为敌方团队
    enemy_team = np.random.choice(df.index, size=6, replace=False).tolist()

    # 优化团队
    optimizer = GeneticOptimizer(df, enemy_team)
    best_teams = optimizer.optimize()

    # 分析结果
    analyzer = TeamAnalyzer(df)
    print(f"\n敌方团队: {enemy_team}")
    for i, (team, win_rate) in enumerate(best_teams):
        print(f"\n=== 推荐队伍 {i + 1} [胜率: {win_rate * 100:.1f}%] ===")
        print(f"成员ID: {team}")
        print("\n属性建议:")
        print(analyzer.generate_advice(team, enemy_team))
        print("\n详细分析:")
        analysis = analyzer.get_team_analysis(team)
        print(f"类型分布: {analysis['type_distribution']}")
        print(f"属性概况: {analysis['stat_summary']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
