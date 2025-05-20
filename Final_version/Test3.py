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



# ==================== Global Configuration ====================
class Config:
    DB_URL = 'mysql+pymysql://root:@localhost:3306/pokemon?charset=utf8mb4'
    STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    TYPE_LIST = ["bug", "dark", "dragon", "electric", "fairy", "fight", "fire", "flying",
                 "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                 "rock", "steel", "water"]  # Keep consistent with database column names
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
    NUM_TEAMS = 5  # Number of teams to generate
    WIN_RATE_SIMULATIONS = 100  # Number of win rate simulations


# ==================== Data Loading and Preprocessing ====================
class PokemonDataLoader:
    def __init__(self):
        self.engine = create_engine(Config.DB_URL)
        self.df = None
        self.feature_columns = []
        self.type_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]

    def load_and_preprocess(self) -> pd.DataFrame:
        """Preprocess to adapt to actual data structure"""
        query = "SELECT * FROM pokemon"
        try:
            self.df = pd.read_sql(query, self.engine).set_index('pokedex_number')
            print(f"Data loaded successfully, total records: {len(self.df)}")
            print("First 3 sample records:\n", self.df.head(3)[['hp', 'attack']])  # Use head() instead of iloc
            print("Feature columns example:", self.feature_columns[:5])
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            raise

        # Intelligent column name matching
        print("Detected columns:", list(self.df.columns))
        
        # Find type columns (case and symbol insensitive)
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
                print(f"Successfully mapped type columns: {rename_map}")
            except Exception as e:
                print(f"Column renaming error: {str(e)}")
                print("Using fallback column names...")
                self.df['original_type1'] = self.df.get(type1_col, 'None')
                self.df['original_type2'] = self.df.get(type2_col, 'None')
            else:
                print("Warning: No type columns detected, creating default columns")
                self.df['original_type1'] = 'None'
                self.df['original_type2'] = 'None'


        # Fill missing values and standardize type columns
        self.df['original_type2'] = self.df['original_type2'].fillna('None')

        # Feature engineering
        self._process_numeric_features()
        self._process_type_features()

        # Build feature matrix (separate feature data from raw data)
        self._build_feature_matrix()
        return self.df

    def _process_type_features(self):
        """Type feature processing optimization"""
        # Use actual existing type columns
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

        # Merge into main dataframe (preserve original type columns)
        self.df = pd.concat([self.df, type_dummies], axis=1)

    def _build_feature_matrix(self):
        """Build feature matrix adapted to actual data"""
        # Define feature columns (adjusted based on actual data, removing skill-related columns)
        # Adjust against-type columns based on actual database column names
        actual_against_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]
        self.feature_columns = (
                Config.STATS +
                [f'type1_{t}' for t in Config.TYPE_LIST] +
                [f'type2_{t}' for t in Config.TYPE_LIST] +
                actual_against_cols
        )

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in self.df:
                self.df[col] = 0.0

        # Create pure numerical feature matrix
        self.feature_df = self.df[self.feature_columns].astype(np.float32)
        # Preserve original data columns
        self.df = self.df[['original_type1', 'original_type2'] + self.feature_columns]

    def _process_numeric_features(self):
        """Process numeric features"""
        stats = Config.STATS
        # Force convert these columns to numeric type and handle possible errors
        self.df[stats] = self.df[stats].apply(pd.to_numeric, errors='coerce')
        # Filter out rows containing NaN
        self.df = self.df.dropna(subset=stats)
        self.df[stats] = (self.df[stats] - self.df[stats].mean()) / self.df[stats].std()
        self.df[stats] = self.df[stats].astype(np.float32)

# ==================== Type Calculation Module ====================
class TypeCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_cols = [f'against_{t.lower()}' for t in Config.TYPE_LIST]
        self._build_type_matrix()

    def _build_type_matrix(self):
        """Build type effectiveness matrix"""
        # Ensure numerical types are correct
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
        # Get battle data (using original type columns)
        attacker = self.tc.df.loc[attacker_idx]
        defender = self.tc.df.loc[defender_idx]

        # Physical/Special attack determination (using standardized attribute values)
        attack_stat = attacker['attack']
        sp_attack_stat = attacker['sp_attack']
        is_physical = attack_stat > sp_attack_stat

        # Get attack/defense values (add smoothing factor to prevent division by zero)
        attack_value = attack_stat if is_physical else sp_attack_stat
        defense_value = defender['defense'] if is_physical else defender['sp_defense']
        defense_value = max(defense_value, 0.1)  # Ensure minimum defense value

        # Get defender's effective types (with type validity check)
        defender_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = defender.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                defender_types.append(t)

        # Get attacker's effective types (for STAB calculation)
        attacker_types = []
        for t_col in ['original_type1', 'original_type2']:
            t = attacker.get(t_col, 'None')
            if t in Config.TYPE_LIST and t != 'None':
                attacker_types.append(t)

        # Calculate type multiplier (with exception handling)
        try:
            type_mult = self.tc.get_effectiveness(attacker_idx, defender_types)
        except IndexError:
            type_mult = 1.0  # Use neutral multiplier in case of exception

        # STAB bonus logic optimization
        if attacker_types:
            move_type = random.choice(attacker_types)
            stab = 1.5 if move_type in attacker_types else 1.0
        else:
            move_type = random.choice(Config.TYPE_LIST)
            stab = 1.0

        # Damage calculation formula optimization (more robust handling)
        safe_attack = np.nan_to_num(attack_value, nan=0.0)  # Handle NaN values
        safe_attack = max(abs(safe_attack), 1e-5)  # Ensure positive value and not less than minimum threshold
        base_damage = 0.5 * (safe_attack ** 1.3)  # Non-linear scaling of attack power
        defense_factor = (defense_value ** 0.8) + 1e-5  # Sub-linear impact of defense power
        raw_damage = base_damage * stab * type_mult / defense_factor

        # Standardized output (sigmoid compression to 0-1 range)
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
        self.gat1 = GATConv(input_dim, 64, heads=4)  # Simplified model
        self.gat2 = GATConv(64 * 4, 32)  # Simplified model
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),  # Change activation function
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.gat1(x, edge_index))  # Change activation function
        x = F.leaky_relu(self.gat2(x, edge_index))  # Change activation function
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
        # Adjust input dimension based on updated feature columns
        input_dim = len(self.df.columns) - 2  # Subtract 'original_type1' and 'original_type2'
        self.model = TeamEvaluator(input_dim=input_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.loss_function = Config.LOSS_FUNCTION

    def generate_training_data(self):
        """Generate training data"""
        samples = []
        for _ in range(Config.TRAIN_SAMPLES):
            team = np.random.choice(self.df.index, size=6, replace=False)
            enemy = np.random.choice(self.df.index, size=6, replace=False)

            # Combine and deduplicate
            combined = list(set(np.concatenate([team, enemy])))
            if len(combined) < 2:  # At least 2 nodes are needed to generate edges
                continue

            # Create graph data
            node_features = torch.tensor(
                self.df.loc[combined].drop(['original_type1', 'original_type2'], axis=1).values,
                dtype=torch.float32
            )
            edge_index = torch.combinations(torch.arange(len(combined)), 2).t()
            data = Data(
                x=node_features,
                edge_index=edge_index,
                y=torch.tensor([self._calc_real_score(team, enemy)], dtype=torch.float32)
            )
            samples.append(data)
        return samples

    def _calc_real_score(self, team, enemy):
        """Calculate real battle score"""
        simulator = BattleSimulator(TypeCalculator(self.df))
        return simulator.team_score(team, enemy)

    def train(self):
        """Train the model"""
        print("Starting to generate training data...")
        try:
            dataset = self.generate_training_data()
            if not dataset:
                raise ValueError("Unable to generate training data, please check data loading and preprocessing")
            print(f"Successfully generated {len(dataset)} training samples")
        except Exception as e:
            print(f"Failed to generate training data: {str(e)}")
            return

        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(Config.TRAIN_EPOCHS):
            total_loss = 0
            print(f"\nEpoch {epoch+1}/{Config.TRAIN_EPOCHS}")
            for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch.to(Config.DEVICE))
                    loss = self.loss_function(outputs, batch.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Output progress every 10 batches
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"Exception occurred during training (Batch {batch_idx}): {str(e)}")
                    continue

            print(f"Epoch {epoch + 1}/{Config.TRAIN_EPOCHS} Loss: {total_loss / len(loader):.4f}")

        torch.save(self.model.state_dict(), Config.MODEL_PATH)
        print(f"Model saved to {Config.MODEL_PATH}")



# ==================== Win Rate Analysis Module ====================
class WinRateAnalyzer:
    def __init__(self, simulator: 'BattleSimulator'):
        self.simulator = simulator

    def calculate_win_rate(self, team: List[int], enemy_team: List[int],
                           num_simulations=Config.WIN_RATE_SIMULATIONS) -> float:
        """Calculate win rate based on specified number of simulations"""
        wins = 0
        for _ in range(num_simulations):
            team_score = 0
            enemy_score = 0

            # Battle simulation: randomly select attacker and defender
            for _ in range(10):  # Simulate 10 random battles each time
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
        """Batch calculate win rates"""
        return [self.calculate_win_rate(team, enemy_team) for team in teams]


# ==================== Team Analysis Module ====================
class TeamAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_cols = [f'type1_{t}' for t in Config.TYPE_LIST] + [f'type2_{t}' for t in Config.TYPE_LIST]

    def get_team_analysis(self, team: List[int]) -> Dict:
        """Get detailed analysis of the team"""
        members = self.df.loc[team]
        return {
            "type_distribution": self._get_type_distribution(members),
            "stat_summary": self._get_stat_summary(members),
            "type_coverage": self._get_type_coverage(members)
        }

    def _get_type_distribution(self, members: pd.DataFrame) -> Dict[str, float]:
        """Get type distribution"""
        type_counts = defaultdict(int)
        for _, row in members.iterrows():
            if row['original_type1'] != 'None':
                type_counts[row['original_type1']] += 1
            if row['original_type2'] != 'None':
                type_counts[row['original_type2']] += 1
        total = sum(type_counts.values())
        return {t: count / total for t, count in type_counts.items()}

    def _get_stat_summary(self, members: pd.DataFrame) -> Dict[str, float]:
        """Get attribute statistics"""
        stats = Config.STATS
        return {
            stat: {
                'mean': members[stat].mean(),
                'max': members[stat].max(),
                'min': members[stat].min()
            } for stat in stats
        }

    def _get_type_coverage(self, members: pd.DataFrame) -> Dict[str, float]:
        """Get type coverage"""
        coverage = defaultdict(float)
        for _, row in members.iterrows():
            for t in Config.TYPE_LIST:
                coverage[t] = max(coverage[t], row[f'against_{t}'])
        return coverage

    def generate_advice(self, team: List[int], enemy_team: List[int]) -> str:
        """Generate team advice"""
        team_analysis = self.get_team_analysis(team)
        enemy_analysis = self.get_team_analysis(enemy_team)

        advice = []
        # Type advice
        team_types = set(team_analysis['type_distribution'].keys())
        enemy_types = set(enemy_analysis['type_distribution'].keys())
        missing_coverage = enemy_types - team_types
        if missing_coverage:
            advice.append(f"Consider adding Pokémon that counter {', '.join(missing_coverage)} types")

        # Attribute advice
        defense_weakness = team_analysis['stat_summary']['defense']['min']
        if defense_weakness < -1.0:
            advice.append("The team has weak physical defense, consider adding defensive Pokémon")

        sp_defense_weakness = team_analysis['stat_summary']['sp_defense']['min']
        if sp_defense_weakness < -1.0:
            advice.append("The team has weak special defense, consider adding special defense Pokémon")

        return "\n".join(advice) if advice else "The team attributes are relatively balanced"


# ==================== Genetic Algorithm Optimizer ====================
class GeneticOptimizer:
    def _try_load_model(self):
        """Try to load the model"""
        if os.path.exists(Config.MODEL_PATH):
            input_dim = len(self.df.columns) - 2  # Subtract 'original_type1' and 'original_type2'
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
        self.unique_teams = set()  # Used to record unique teams

    def _analyze_enemy_types(self) -> np.ndarray:
        """Analyze enemy type distribution"""
        from __main__ import Config  # Add reference to Config class
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
        """Initialize population: generate random Pokémon teams"""
        population = []
        for _ in range(Config.POP_SIZE):
            # Ensure teams are unique and do not include enemy members
            team = np.random.choice(self.pool, size=6, replace=False).tolist()
            while frozenset(team) in self.unique_teams:
                team = np.random.choice(self.pool, size=6, replace=False).tolist()
            population.append(team)
        return population

    def _fitness(self, team: List[int]) -> float:
        """Calculate team fitness: combine type advantage and model prediction"""
        from __main__ import Config  # Add reference to Config class
        
        # Type score: effectiveness against enemy types
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
        
        # Attribute score: team attribute balance
        stats = self.df.loc[team][Config.STATS].values
        stat_score = np.mean(stats) - 0.5 * np.std(stats)  # Encourage high mean and low variance
        
        # Model prediction score (if model exists)
        model_score = 0.0
        if self.model:
            with torch.no_grad():
                # Dynamically generate node indices
                team_features = self.df.loc[team].drop(['original_type1', 'original_type2'], axis=1)
                num_nodes = len(team_features)
                if num_nodes < 2:  # Handle special cases
                    return 0.0
                
                # Generate valid edge indices
                edge_index = torch.combinations(torch.arange(num_nodes), 2).t()
                data = Data(
                    x=torch.tensor(team_features.values, dtype=torch.float32),
                    edge_index=edge_index
                )
                model_score = self.model(data.to(Config.DEVICE)).item()
        
        # Combined score weights
        return 0.4 * type_score + 0.3 * stat_score + 0.3 * model_score

    def optimize(self) -> List[Tuple[List[int], float]]:
        """Optimization main loop, return multiple optimal teams"""
        population = self._initialize_population()
        best_teams = []

        with tqdm(total=Config.GENERATIONS, desc="Optimizing") as pbar:
            for generation in range(Config.GENERATIONS):
                # Evaluate fitness
                scores = [self._fitness(t) for t in population]

                # Update best teams (retain top 10%)
                sorted_indices = np.argsort(scores)[::-1]
                for idx in sorted_indices[:Config.POP_SIZE // 10]:
                    team = population[idx]
                    team_hash = frozenset(team)
                    if team_hash not in self.unique_teams:
                        self.unique_teams.add(team_hash)
                        best_teams.append((team, scores[idx]))

                # Selection (tournament selection)
                selected = self._tournament_selection(population, scores)

                # Crossover and mutation
                population = self._crossover_and_mutate(selected, generation)
                pbar.update(1)

        # Post-processing: filter and calculate real win rates
        return self._post_process(best_teams)

    def _post_process(self, candidates: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
        """Post-processing: deduplicate and calculate real win rates"""
        # Deduplication
        unique = {}
        for team, score in candidates:
            key = frozenset(team)
            if key not in unique or score > unique[key][1]:
                unique[key] = (team, score)

        # Take the top N candidates
        sorted_teams = sorted(unique.values(), key=lambda x: x[1], reverse=True)[:Config.NUM_TEAMS * 2]

        # Calculate real win rates
        final_teams = []
        for team, _ in sorted_teams:
            win_rate = self.win_analyzer.calculate_win_rate(team, self.enemy_team)
            final_teams.append((team, win_rate))

        # Sort by win rate and return the top N
        return sorted(final_teams, key=lambda x: x[1], reverse=True)[:Config.NUM_TEAMS]

    def _tournament_selection(self, population, scores):
        """Tournament selection"""
        selected = []
        for _ in range(Config.POP_SIZE):
            candidates = random.sample(list(zip(population, scores)), 3)
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def _crossover_and_mutate(self, selected, generation):
        """Improved crossover and mutation operations"""
        new_pop = []
        
        # Multi-point crossover
        for i in range(0, len(selected), 2):
            if i + 1 >= len(selected):
                break
            parent1, parent2 = selected[i], selected[i + 1]
            
            # Generate two crossover points
            cross_points = sorted(random.sample(range(1, 6), 2))
            child1 = parent1[:cross_points[0]] + parent2[cross_points[0]:cross_points[1]] + parent1[cross_points[1]:]
            child2 = parent2[:cross_points[0]] + parent1[cross_points[0]:cross_points[1]] + parent2[cross_points[1]:]
            
            # Remove duplicate IDs and fill up
            child1 = list(dict.fromkeys(child1))
            while len(child1) < 6:
                available = [p for p in self.pool if p not in child1]
                if not available:
                    break  # Prevent infinite loop if pool is exhausted
                child1.append(random.choice(available))
            child1 = child1[:6]

            child2 = list(dict.fromkeys(child2))
            while len(child2) < 6:
                available = [p for p in self.pool if p not in child2]
                if not available:
                    break
                child2.append(random.choice(available))
            child2 = child2[:6]
            
            # Adaptive mutation
            new_pop.extend([self._mutate(child1, generation), self._mutate(child2, generation)])
            
        return new_pop

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Improved crossover operation to maintain diversity"""
        common = set(parent1) & set(parent2)
        unique1 = [p for p in parent1 if p not in common]
        unique2 = [p for p in parent2 if p not in common]

        # Retain common elements, randomly supplement unique elements
        child1 = list(common) + random.sample(unique1 + unique2, 6 - len(common))
        child2 = list(common) + random.sample(unique1 + unique2, 6 - len(common))

        # Ensure no duplicates
        child1 = list(dict.fromkeys(child1))[:6]
        child2 = list(dict.fromkeys(child2))[:6]
        return child1, child2

    def _mutate(self, team: List[int], generation: int) -> List[int]:
        """Improved mutation operation, considering type balance"""
        for i in range(len(team)):
            if random.random() < Config.MUTATION_RATE:
                # Select new members based on type complementarity
                current_types = self._get_team_types(team)
                candidate_pool = [p for p in self.pool if self._is_type_complementary(p, current_types)]
                if candidate_pool:
                    team[i] = random.choice(candidate_pool)
        return team

    def _get_team_types(self, team: List[int]) -> List[str]:
        """Get all types of the team"""
        types = []
        for idx in team:
            types.append(self.df.loc[idx, 'original_type1'])
            if self.df.loc[idx, 'original_type2'] != 'None':
                types.append(self.df.loc[idx, 'original_type2'])
        return list(set(types))

    def _is_type_complementary(self, candidate: int, current_types: List[str]) -> bool:
        """Determine whether the candidate complements the team's types"""
        candidate_types = [
            self.df.loc[candidate, 'original_type1'],
            self.df.loc[candidate, 'original_type2']
        ]
        candidate_types = [t for t in candidate_types if t != 'None']
        return any(t not in current_types for t in candidate_types)


# ==================== Main Function Improvements ====================
def main():
    # Load and preprocess data
    loader = PokemonDataLoader()
    df = loader.load_and_preprocess()

    # Train the model (if it does not exist)
    if not os.path.exists(Config.MODEL_PATH):
        trainer = ModelTrainer(df)
        trainer.train()

    # Randomly select six Pokémon from the database as the enemy team
    enemy_team = np.random.choice(df.index, size=6, replace=False).tolist()

    # Optimize the team
    optimizer = GeneticOptimizer(df, enemy_team)
    best_teams = optimizer.optimize()

    # Analyze the results
    analyzer = TeamAnalyzer(df)
    print(f"\nEnemy team: {enemy_team}")
    for i, (team, win_rate) in enumerate(best_teams):
        print(f"\n=== Recommended Team {i + 1} [Win Rate: {win_rate * 100:.1f}%] ===")
        print(f"Member IDs: {team}")
        print("\nAttribute Suggestions:")
        print(analyzer.generate_advice(team, enemy_team))
        print("\nDetailed Analysis:")
        analysis = analyzer.get_team_analysis(team)
        print(f"Type Distribution: {analysis['type_distribution']}")
        print(f"Attribute Summary: {analysis['stat_summary']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
