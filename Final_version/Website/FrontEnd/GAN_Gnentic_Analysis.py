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


# ==================== Global Configuration ====================
class Config:
    DB_URL ='mysql+pymysql://root:@localhost:3306/poke?charset=utf8mb4'
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


# ==================== Data Loading and Preprocessing ====================
class PokemonDataLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.feature_columns = []
        self.type_cols = [f'against_{t}' for t in Config.TYPE_LIST]

    def load_and_preprocess(self) -> pd.DataFrame:
        """Preprocessing adapted to the actual data structure"""
        query = "SELECT * FROM pokemon"
        self.df = pd.read_sql(query, self.engine)

        # Rename columns for compatibility (actual column names in data are type1/type2)
        self.df = self.df.rename(columns={
            'type1': 'original_type1',
            'type2': 'original_type2'
        })

        # Fill null values and standardize type columns
        self.df['original_type2'] = self.df['original_type2'].fillna('None')

        # Feature engineering
        self._process_numeric_features()
        self._process_type_features()

        # Build feature matrix (separate feature data and original data)
        self._build_feature_matrix()
        return self.df

    def _process_type_features(self):
        """Optimized type feature processing"""
        # Use existing type columns
        type_df = self.df[['original_type1', 'original_type2']].copy()

        # Filter invalid type values
        valid_types = set(Config.TYPE_LIST + ['None'])
        type_df['original_type1'] = type_df['original_type1'].where(
            type_df['original_type1'].isin(valid_types), 'None')
        type_df['original_type2'] = type_df['original_type2'].where(
            type_df['original_type2'].isin(valid_types), 'None')

        # Generate dummy variables
        type_dummies = pd.get_dummies(
            type_df,
            prefix=['type1', 'type2'],
            columns=['original_type1', 'original_type2'],
            dtype=np.float32
        )

        # Merge into the main dataframe (keep original type columns)
        self.df = pd.concat([self.df, type_dummies], axis=1)

    def _build_feature_matrix(self):
        """Build a feature matrix adapted to the actual data"""
        # Define feature columns (adjust according to actual data, remove move-related columns)
        self.feature_columns = (
                Config.STATS +
                [f'type1_{t}' for t in Config.TYPE_LIST] +
                [f'type2_{t}' for t in Config.TYPE_LIST] +
                [f'against_{t}' for t in Config.TYPE_LIST]
        )

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in self.df:
                self.df[col] = 0.0

        # Create a pure numerical feature matrix
        self.feature_df = self.df[self.feature_columns].astype(np.float32)
        # Keep original data columns
        self.df = self.df[['original_type1', 'original_type2'] + self.feature_columns]

    def _process_numeric_features(self):
        """Process numerical features"""
        stats = Config.STATS
        # Force these columns to be numerical and handle possible error values
        self.df[stats] = self.df[stats].apply(pd.to_numeric, errors='coerce')
        # Filter out rows containing NaN
        self.df = self.df.dropna(subset=stats)
        self.df[stats] = (self.df[stats] - self.df[stats].mean()) / self.df[stats].std()
        self.df[stats] = self.df[stats].astype(np.float32)


# ==================== Type Calculation Module ====================
class TypeCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_cols = [f'against_{t}' for t in Config.TYPE_LIST]
        self._build_type_matrix()

    def _build_type_matrix(self):
        """Build a type effectiveness matrix"""
        # Ensure correct numerical types
        self.type_matrix = self.df[self.type_cols].values.astype(np.float32)

    def get_effectiveness(self, attacker_idx: int, defender_types: List[str]) -> float:
        """Get the total multiplier of the attacker against the defender's types"""
        effectiveness = 1.0
        for t in defender_types:
            if t == 'None':
                continue
            col_idx = Config.TYPE_LIST.index(t)
            effectiveness *= self.type_matrix[attacker_idx, col_idx]
        return effectiveness


# ==================== Battle Simulator ====================
class BattleSimulator:
    def __init__(self, type_calculator: TypeCalculator):
        self.tc = type_calculator

    def evaluate_matchup(self, attacker_idx: int, defender_idx: int) -> float:
        """Optimized battle evaluation function"""
        # Get battle data (use original type columns)
        attacker = self.tc.df.loc[attacker_idx]
        defender = self.tc.df.loc[defender_idx]

        # Physical/special attack judgment (use standardized attribute values)
        attack_stat = attacker['attack']
        sp_attack_stat = attacker['sp_attack']
        is_physical = attack_stat > sp_attack_stat

        # Get attack/defense values (add a smoothing factor to prevent division by zero)
        attack_value = attack_stat if is_physical else sp_attack_stat
        defense_value = defender['defense'] if is_physical else defender['sp_defense']
        defense_value = max(defense_value, 0.1)  # Ensure minimum defense value

        # Get the defender's effective types (with type effectiveness verification)
        defender_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = defender.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                defender_types.append(t)

        # Get the attacker's effective types (for STAB calculation)
        attacker_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = attacker.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                attacker_types.append(t)

        # Calculate type multiplier (with exception handling)
        try:
            type_mult = self.tc.get_effectiveness(attacker_idx, defender_types)
        except IndexError:
            type_mult = 1.0  # Use a neutral multiplier in case of an exception

        # Optimized STAB bonus logic
        if attacker_types:
            move_type = random.choice(attacker_types)
            stab = 1.5 if move_type in attacker_types else 1.0
        else:
            move_type = random.choice(Config.TYPE_LIST)
            stab = 1.0

        # Optimized damage calculation formula (more robust handling)
        safe_attack = np.nan_to_num(attack_value, nan=0.0)  # Handle NaN values
        safe_attack = max(abs(safe_attack), 1e-5)  # Ensure positive value and not less than the minimum threshold
        base_damage = 0.5 * (safe_attack ** 1.3)  # Non-linear scaling of attack power
        defense_factor = (defense_value ** 0.8) + 1e-5  # Sub-linear influence of defense
        raw_damage = base_damage * stab * type_mult / defense_factor

        # Normalize output (compress to the 0-1 range using sigmoid)
        normalized_damage = 1 / (1 + np.exp(-raw_damage / 50 + 3))
        return np.clip(normalized_damage, 0.0, 1.0)

    def team_score(self, team: List[int], enemy_team: List[int]) -> float:
        """Team score calculation"""
        total = 0.0
        for a in team:
            for e in enemy_team:
                total += self.evaluate_matchup(a, e)
        return total / (len(team) * len(enemy_team))


# ==================== Neural Network Model ====================
class TeamEvaluator(nn.Module):
    """Team evaluation model"""

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
        # Adjust input dimension according to the updated number of feature columns
        input_dim = len(self.df.columns) - 2  # Subtract 'original_type1' and 'original_type2'
        self.model = TeamEvaluator(input_dim=input_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def generate_training_data(self):
        """Generate training data"""
        samples = []
        for _ in range(Config.TRAIN_SAMPLES):
            team = np.random.choice(self.df.index, size=6, replace=False)
            enemy = np.random.choice(self.df.index, size=6, replace=False)

            # Create graph data
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
        """Calculate the real battle score"""
        simulator = BattleSimulator(TypeCalculator(self.df))
        return simulator.team_score(team, enemy)

    def train(self):
        """Train the model"""
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
        print(f"Model saved to {Config.MODEL_PATH}")


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
        if os.path.exists(Config.MODEL_PATH):
            input_dim = len(self.df.columns) - 2  
            model = TeamEvaluator(input_dim=input_dim)
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            return model.eval()
        return None

    def _analyze_enemy_types(self) -> np.ndarray:
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
        population = []
        pool_array = np.array(self.pool)

        for _ in range(Config.POP_SIZE):
            scores = self.tc.type_matrix[self.pool] @ self.enemy_types
            if scores.sum() == 0:
                probs = np.ones(len(pool_array)) / len(pool_array)
            else:
                probs = scores / scores.sum()
            selected = np.random.choice(pool_array, 6, p=probs, replace=False)
            population.append(list(selected))
        return population

    def _fitness(self, team: List[int]) -> float:
        if self.model:
            return self._model_predict(team)
        return self.simulator.team_score(team, self.enemy_team)

    def _model_predict(self, team: List[int]) -> float:
        data = self._create_graph_data(team)
        with torch.no_grad():
            return self.model(data.to(Config.DEVICE)).item()

    def _create_graph_data(self, team: List[int]) -> Data:
        node_features = torch.tensor(
            self.df.loc[team + self.enemy_team].drop(['original_type1', 'original_type2'], axis=1).values,
            dtype=torch.float32
        )
        edge_index = torch.combinations(torch.arange(node_features.size(0)), 2).t()
        return Data(x=node_features, edge_index=edge_index)

    def optimize(self) -> Tuple[List[int], float]:
        """Optimization main loop"""
        population = self._initialize_population()
        best_team, best_score = None, -np.inf

        with tqdm(total=Config.GENERATIONS, desc="Optimizing") as pbar:
            for _ in range(Config.GENERATIONS):
                # Evaluate fitness
                scores = [self._fitness(t) for t in population]

                # Update the best team
                current_best_idx = np.argmax(scores)
                if scores[current_best_idx] > best_score:
                    best_score = scores[current_best_idx]
                    best_team = population[current_best_idx]

                # Selection (tournament selection)
                selected = []
                for _ in range(Config.POP_SIZE):
                    candidates = random.sample(population, 3)
                    winner = max(candidates, key=lambda t: self._fitness(t))
                    selected.append(winner)

                # Crossover and mutation
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
        """Crossover operation"""
        crossover_point = random.randint(1, 5)
        child1 = parent1[:crossover_point] + [p for p in parent2 if p not in parent1[:crossover_point]][:6 - crossover_point]
        child2 = parent2[:crossover_point] + [p for p in parent1 if p not in parent2[:crossover_point]][:6 - crossover_point]
        return child1, child2

    def _mutate(self, team: List[int]) -> List[int]:
        """Mutation operation"""
        for i in range(len(team)):
            if random.random() < Config.MUTATION_RATE:
                new_pokemon = random.choice(self.pool)
                while new_pokemon in team:
                    new_pokemon = random.choice(self.pool)
                team[i] = new_pokemon
        return team


# Main function
def main():
    # Load and preprocess data
    loader = PokemonDataLoader()
    df = loader.load_and_preprocess()
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)
    engine = create_engine(Config.DB_URL)

    # Query rows in the battle_records table where recommended_pokemon_1 is null
    with engine.connect() as conn:
        select_query = text("SELECT id, selected_pokemon_1, selected_pokemon_2, selected_pokemon_3, "
                            "selected_pokemon_4, selected_pokemon_5, selected_pokemon_6 "
                            "FROM battle_records WHERE recommended_pokemon_1 IS NULL")
        results = conn.execute(select_query).fetchall()

        for row in results:
            row_id = row[0]
            enemy_team = [row[1], row[2], row[3], row[4], row[5], row[6]]

            # Optimize the team
            optimizer = GeneticOptimizer(df, enemy_team)
            best_team, best_score = optimizer.optimize()

            print(f"Enemy team indices: {enemy_team}")
            print(f"Recommended team indices: {best_team}")
            print(f"Best score: {best_score}")

            # Update the recommended team information in the battle_records table
            update_query = text("UPDATE battle_records "
                                "SET recommended_pokemon_1 = :recommended_pokemon_1, "
                                "recommended_pokemon_2 = :recommended_pokemon_2, "
                                "recommended_pokemon_3 = :recommended_pokemon_3, "
                                "recommended_pokemon_4 = :recommended_pokemon_4, "
                                "recommended_pokemon_5 = :recommended_pokemon_5, "
                                "recommended_pokemon_6 = :recommended_pokemon_6 "
                                "WHERE id = :id")
            conn.execute(update_query, {
                "recommended_pokemon_1": best_team[0],
                "recommended_pokemon_2": best_team[1],
                "recommended_pokemon_3": best_team[2],
                "recommended_pokemon_4": best_team[3],
                "recommended_pokemon_5": best_team[4],
                "recommended_pokemon_6": best_team[5],
                "id": row_id
            })
        conn.commit()


if __name__ == "__main__":
    main()
    