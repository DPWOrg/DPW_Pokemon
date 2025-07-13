# Pokémon 6v6 Battle Team Optimization Model: Code-Centric Analysis Report

---

## **1. Win Rate Analysis Module**

### **1.1 Monte Carlo Simulation**
**Formula**: 

$$\text{WinRate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{TeamScore}_i > \text{EnemyScore}_i)$$

**Code Implementation**:  
```python
def calculate_win_rate(self, team: List[int], enemy_team: List[int], num_simulations=1000) -> float:
    wins = 0
    for _ in range(num_simulations):
        team_score, enemy_score = 0, 0
        # Simulate 10 random battles per iteration
        for _ in range(10):  
            attacker = random.choice(team + enemy_team)
            defender = random.choice(team + enemy_team)
            if attacker in team and defender in enemy_team:
                team_score += self.simulator.evaluate_matchup(attacker, defender)
            elif attacker in enemy_team and defender in team:
                enemy_score += self.simulator.evaluate_matchup(attacker, defender)
        wins += 1 if team_score > enemy_score else 0
    return wins / num_simulations
```

#### **Explanation**:
1. **Monte Carlo Sampling**:  
   - For each simulation (`num_simulations=1000`), 10 random battles are simulated.  
   - `attacker` and `defender` are randomly selected from both teams.  
2. **Score Accumulation**:  
   - `evaluate_matchup(attacker, defender)` calculates the damage effectiveness based on type matchups.  
   - Scores are accumulated based on the attacker-defender affiliation.  
3. **Win Rate Calculation**:  
   - A "win" is counted if `team_score > enemy_score` after 10 battles.  
   - Final win rate is the ratio of wins to total simulations.

---
## **3. Genetic Algorithm Optimization**

### **3.1 Fitness Function**
**Formula**:  
$$\text{Fitness} = 0.4 \times \text{TypeScore} + 0.3 \times (\mu_{\text{stats}} - 0.5\sigma_{\text{stats}}) + 0.3 \times \text{ModelScore}$$ 
**Code Implementation**:  
```python
def _fitness(self, team: List[int]) -> float:
    # TypeScore calculation
    type_score = 0.0
    team_types = self._get_team_types(team)
    enemy_main_type = Config.TYPE_LIST[self.enemy_types.argmax()]
    for t in team_types:
        if t in Config.TYPE_LIST:
            type_score += self.tc.get_effectiveness(
                Config.TYPE_LIST.index(t), 
                [enemy_main_type]
            )
    
    # StatScore calculation
    stats = self.df.loc[team][Config.STATS].values
    stat_score = np.mean(stats) - 0.5 * np.std(stats)
    
    # ModelScore calculation (if model exists)
    model_score = 0.0
    if self.model:
        team_features = self.df.loc[team].drop(['original_type1', 'original_type2'], axis=1)
        data = Data(
            x=torch.tensor(team_features.values, dtype=torch.float32),
            edge_index=torch.combinations(torch.arange(len(team_features)), 2).t()
        )
        model_score = self.model(data).item()
    
    return 0.4 * type_score + 0.3 * stat_score + 0.3 * model_score
```

#### **Explanation**:
1. **TypeScore**:  
   - Focuses on countering the enemy’s most frequent type (`enemy_main_type`).  
   - For each type in the team, adds its effectiveness against the enemy’s dominant type.  
2. **StatScore**:  
   - Rewards teams with high average stats and low variance.  
3. **ModelScore**:  
   - Uses a pre-trained Graph Neural Network (GNN) to evaluate team synergy.  
   - Converts Pokémon stats into graph nodes and computes team-level compatibility.  

### **3.2 Crossover Operation**
**Code Implementation**:  
```python
def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    common = set(parent1) & set(parent2)
    unique1 = [p for p in parent1 if p not in common]
    unique2 = [p for p in parent2 if p not in common]
    child1 = list(common) + random.sample(unique1 + unique2, 6 - len(common))
    child2 = list(common) + random.sample(unique1 + unique2, 6 - len(common))
    return child1[:6], child2[:6]
```

#### **Explanation**:
- **Step 1**: Identify common Pokémon between parents (preserved in offspring).  
- **Step 2**: Combine unique Pokémon from both parents into a pool.  
- **Step 3**: Randomly sample from the pool to fill remaining slots (6 total).  
- **Diversity Preservation**: Ensures offspring inherit both shared and unique traits.  

### **3.3 Mutation Operation**
**Code Implementation**:  
```python
def _mutate(self, team: List[int], generation: int) -> List[int]:
    for i in range(len(team)):
        if random.random() < Config.MUTATION_RATE:
            current_types = self._get_team_types(team)
            candidate_pool = [
                p for p in self.pool 
                if self._is_type_complementary(p, current_types)
            ]
            if candidate_pool:
                team[i] = random.choice(candidate_pool)
    return team
```

#### **Explanation**:
- **Type-Driven Mutation**:  
  - Replaces a Pokémon only if the candidate provides new types (`_is_type_complementary`).  
  - Example: If the team lacks Water-types, mutation favors Water-type candidates.  
- **Adaptive Rate**:  
  - `Config.MUTATION_RATE` controls exploration vs. exploitation (default: 5-10%).  

---

## **4. Neural Network Integration**

### **4.1 GNN Architecture**
**Code Implementation**:  
```python
class TeamEvaluator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = Linear(64, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index))
        return torch.sigmoid(self.fc(x.mean(dim=0)))
```

#### **Explanation**:
- **Graph Construction**:  
  - **Nodes**: Pokémon features (stats, types).  
  - **Edges**: All pairwise connections within the team.  
- **Layers**:  
  1. **GCNConv**: Aggregates neighbor information to capture synergies.  
  2. **Dropout**: Prevents overfitting (50% dropout rate).  
  3. **Linear Layer**: Outputs a team compatibility score (0–1).  

---

## **5. Key Formulas**

### **5.1 Type Effectiveness Matrix**
$$M_{A,D} = \prod_{t_A \in A} \prod_{t_D \in D} \text{Effectiveness}(t_A, t_D)$$
- $t_A$: Attacker’s type(s).  
- $t_D$: Defender’s type(s).  
- Example: Fire vs. Grass/Poison = \( 2.0 \times 2.0 = 4.0 \).  

### **5.2 Stat Balance Metric**
$$\text{Balance} = \mu_{\text{stats}} - 0.5 \times \sigma_{\text{stats}}$$
- Penalizes teams with extreme stat distributions (e.g., all high Attack, low Defense).  

---

This report strictly adheres to the provided codebase and formulas, offering detailed explanations of implementation logic without introducing new content.
