import json
import random
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from Test2_2 import Config, PokemonDataLoader, TypeCalculator, BattleSimulator

class AdvancedGeneticOptimizer:
    def __init__(self, df: pd.DataFrame, enemy_team: List[int]):
        self.df = df
        self.enemy_team = enemy_team
        self.tc = TypeCalculator(df)
        self.simulator = BattleSimulator(self.tc)
        self.pool = [i for i in df.index if i not in enemy_team]
        
        # 多目标优化参数
        self.pop_size = 500
        self.generations = 100
        self.fronts = []
        
    def _nsga2_selection(self, population: List[Dict]) -> List[Dict]:
        """NSGA-II选择机制"""
        # 非支配排序
        fronts = self._non_dominated_sort(population)
        
        # 拥挤度计算
        for front in fronts:
            self._calculate_crowding_distance(front)
            
        # 选择新种群
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(front)
            else:
                front.sort(key=lambda x: x['crowding'], reverse=True)
                new_pop.extend(front[:self.pop_size - len(new_pop)])
                break
        return new_pop
    
    def _non_dominated_sort(self, population: List[Dict]) -> List[List[Dict]]:
        """非支配排序"""
        fronts = [[]]
        for ind in population:
            ind['dominated'] = []
            ind['dom_count'] = 0
            for other in population:
                if self._dominates(ind, other):
                    ind['dominated'].append(other)
                elif self._dominates(other, ind):
                    ind['dom_count'] += 1
            if ind['dom_count'] == 0:
                fronts[0].append(ind)
                
        i = 0
        while fronts[i]:
            next_front = []
            for ind in fronts[i]:
                for dominated in ind['dominated']:
                    dominated['dom_count'] -= 1
                    if dominated['dom_count'] == 0:
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)
        return fronts
    
    def _dominates(self, a: Dict, b: Dict) -> bool:
        """多目标支配关系"""
        return (a['score'] >= b['score'] and 
                a['diversity'] >= b['diversity'] and
                a['type_coverage'] >= b['type_coverage'] and
                (a['score'] > b['score'] or 
                 a['diversity'] > b['diversity'] or
                 a['type_coverage'] > b['type_coverage']))
    
    def _calculate_crowding_distance(self, front: List[Dict]):
        """拥挤度计算"""
        for ind in front:
            ind['crowding'] = 0.0
            
        for m in ['score', 'diversity', 'type_coverage']:
            front.sort(key=lambda x: x[m])
            front[0]['crowding'] = float('inf')
            front[-1]['crowding'] = float('inf')
            norm = front[-1][m] - front[0][m]
            if norm == 0: continue
            for i in range(1, len(front)-1):
                front[i]['crowding'] += (front[i+1][m] - front[i-1][m]) / norm

class WinRateAnalyzer:
    def __init__(self, simulator: BattleSimulator):
        self.simulator = simulator
        
    def monte_carlo_simulation(self, team: List[int], enemy: List[int], n=1000) -> float:
        """蒙特卡洛胜率模拟"""
        wins = 0
        for _ in range(n):
            our_score = self.simulator.team_score(team, enemy)
            their_score = self.simulator.team_score(enemy, team)
            if our_score > their_score:
                wins += 1
        return wins / n
    
class TeamAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_team_analysis(self, team: List[int]) -> Dict:
        """团队综合分析"""
        types = defaultdict(int)
        stats_sum = defaultdict(float)
        
        for idx in team:
            row = self.df.iloc[idx]
            # 类型统计
            if row['original_type1'] != 'None':
                types[row['original_type1']] += 1
            if row['original_type2'] != 'None':
                types[row['original_type2']] += 1
                
            # 属性统计
            for stat in Config.STATS:
                stats_sum[stat] += row[stat]
                
        # 类型覆盖率计算
        coverage = len(types) / len(Config.TYPE_LIST)
        
        # 属性平衡性分析
        stat_values = list(stats_sum.values())
        cv = np.std(stat_values) / np.mean(stat_values)  # 变异系数
        
        return {
            'type_coverage': coverage,
            'stat_balance': 1 - cv,  # 平衡性指标
            'type_distribution': dict(types),
            'average_stats': {k: v/6 for k,v in stats_sum.items()}
        }

def enhanced_main():
    # 初始化数据
    loader = PokemonDataLoader()
    df = loader.load_and_preprocess()
    enemy_team = np.random.choice(df.index, size=6, replace=False).tolist()
    
    # 初始化优化器
    optimizer = AdvancedGeneticOptimizer(df, enemy_team)
    optimizer.initialize_population()
    
    # 优化配置
    optimizer = AdvancedGeneticOptimizer(df, enemy_team)
    analyzer = TeamAnalyzer(df)
    winrate = WinRateAnalyzer(BattleSimulator(TypeCalculator(df)))
    
    # 运行优化
    final_teams = optimizer.optimize()
    
    # 结果处理
    results = []
    for team in final_teams[:5]:  # 取前5个最优团队
        analysis = analyzer.get_team_analysis(team)
        wr = winrate.monte_carlo_simulation(team, enemy_team)
        
        if wr < 0.7:
            continue  # 过滤胜率不足的团队
            
        results.append({
            "team": team,
            "win_rate": wr,
            "type_suggestions": _generate_type_suggestions(analysis['type_distribution']),
            "stat_suggestions": _generate_stat_suggestions(analysis['average_stats']),
            "analysis": analysis
        })
    
    # 保存结果
    with open('team_recommendations.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"生成{len(results)}个有效团队，结果已保存至team_recommendations.json")

def _generate_type_suggestions(type_dist: Dict) -> List[str]:
    """生成类型建议"""
    """生成类型建议"""
    suggestions = []
    # 找出覆盖率低的类型
    all_types = set(Config.TYPE_LIST)
    covered = set(type_dist.keys())
    missing = all_types - covered
    
    if missing:
        suggestions.append(f"推荐补充 {', '.join(missing)} 类型宝可梦")
        
    # 检查类型重复
    over_rep = [t for t,c in type_dist.items() if c > 2]
    if over_rep:
        suggestions.append(f"减少 {', '.join(over_rep)} 类型重复")
        
    return suggestions

def _generate_stat_suggestions(stats: Dict) -> List[str]:
    """生成属性建议"""
    """生成属性建议"""
    suggestions = []
    sorted_stats = sorted(stats.items(), key=lambda x: x[1])
    
    # 最低的三项属性
    weak = [k for k,v in sorted_stats[:3]]
    if weak:
        suggestions.append(f"强化 {', '.join(weak)} 属性")
        
    # 检查属性平衡
    cv = np.std(list(stats.values())) / np.mean(list(stats.values()))
    if cv > 0.15:
        suggestions.append("团队属性不够均衡，建议平衡发展")
        
    return suggestions

if __name__ == "__main__":
    enhanced_main()
