#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pokemon Types Analysis
This script analyzes Pokemon type data from CSV files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_pokemon_data():
    """Load all Pokemon type data"""
    # Get absolute path of the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set data directory
    data_dir = current_dir
    
    print(f"Loading data from directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist")
        return {}
    
    # List all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"CSV files found: {csv_files}")
    
    if not csv_files:
        print("No CSV files found. Please ensure data files are in the correct directory.")
        return {}
    
    # Read data for each type
    type_data = {}
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            # Extract type name (remove .csv extension)
            type_name = os.path.splitext(csv_file)[0]
            df = pd.read_csv(file_path)
            
            # Add type column to identify Pokemon type
            df['type'] = type_name
            
            type_data[type_name] = df
            print(f"Successfully loaded {type_name} type Pokemon data, {len(df)} records")
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
    
    return type_data

def analyze_stats_by_type(type_data):
    """Analyze statistics for different Pokemon types"""
    if not type_data:
        print("No data available for analysis")
        return
    
    # Create a DataFrame combining all data
    all_pokemon = pd.concat(type_data.values(), ignore_index=True)
    
    # Confirm data structure
    print("\nData structure example:")
    print(all_pokemon.head())
    
    # Check column names to adjust analysis code
    print("\nData columns:")
    print(all_pokemon.columns.tolist())
    
    # Define expected stat columns and their possible alternatives
    stat_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total_stats']
    
    column_mapping = {
        'HP': ['HP', 'hp', 'Hit_Points'],
        'Attack': ['Attack', 'attack', 'Atk'],
        'Defense': ['Defense', 'defense', 'Def'],
        'Sp. Atk': ['Sp. Atk', 'sp_atk', 'Special_Attack', 'Sp_Atk', 'sp_attack'],
        'Sp. Def': ['Sp. Def', 'sp_def', 'Special_Defense', 'Sp_Def', 'sp_defense'],
        'Speed': ['Speed', 'speed', 'Spd'],
        'Total_stats': ['Total_stats', 'total', 'Total', 'total_stats']
    }
    
    actual_columns = all_pokemon.columns.tolist()
    adjusted_stat_columns = []
    stat_mapping = {}
    
    for std_name, alternatives in column_mapping.items():
        found = False
        for alt in alternatives:
            if alt in actual_columns:
                adjusted_stat_columns.append(alt)
                stat_mapping[std_name] = alt
                found = True
                break
        if not found:
            print(f"Warning: Could not find '{std_name}' column or its alternatives")
    
    if not adjusted_stat_columns:
        print("Error: Could not find any stat-related columns for analysis")
        return
    
    print(f"\nUsing the following columns for analysis: {adjusted_stat_columns}")
    
    # Analyze average stats for each type
    print("\nAverage stats by Pokemon type:")
    type_avg_stats = all_pokemon.groupby('type')[adjusted_stat_columns].mean()
    print(type_avg_stats)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save average stats data to CSV
    avg_stats_file = os.path.join(output_dir, 'average_stats_by_type.csv')
    type_avg_stats.to_csv(avg_stats_file)
    print(f"Average stats data saved to: {avg_stats_file}")
    
    # Define types list from the groupby result
    types = type_avg_stats.index.tolist()
    
    # 1. Visualize average total stats by type (bar chart)
    plt.figure(figsize=(12, 8))
    
    total_col = [col for col in adjusted_stat_columns if 'total' in col.lower()][0] if any('total' in col.lower() for col in adjusted_stat_columns) else None
    
    if total_col:
        # Sort by total stats descending
        sorted_data = type_avg_stats.sort_values(by=total_col, ascending=False)
        ax = sorted_data[total_col].plot(kind='bar', color='teal')
        plt.title('Average Total Stats by Pokemon Type', fontsize=16)
        plt.xlabel('Type', fontsize=14)
        plt.ylabel('Average Total Stats', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(sorted_data[total_col]):
            ax.text(i, v + 5, f'{v:.1f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_total_stats_by_type.png'), dpi=300)
        plt.close()
    
    # 2. Visualize the six base stats by type (radar chart)
    # Use only the six base stats, not the total
    base_stats = [col for col in adjusted_stat_columns if 'total' not in col.lower()]
    
    if len(base_stats) >= 5:  # Need at least several basic stats for radar chart
        # Create radar chart for each type
        # Note: types is already defined above
        
        # Create color mapping
        colors = plt.cm.jet(np.linspace(0, 1, len(types)))
        
        # Draw radar chart for all types
        plt.figure(figsize=(12, 10))
        
        # Create radar chart coordinates
        angles = np.linspace(0, 2*np.pi, len(base_stats), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        ax = plt.subplot(111, polar=True)
        
        # Plot data for each type
        for i, type_name in enumerate(types):
            values = type_avg_stats.loc[type_name, base_stats].tolist()
            values += values[:1]  # Close the plot
            
            ax.plot(angles, values, 'o-', linewidth=2, label=type_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set radar chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([stat_mapping.get(col, col) for col in base_stats])
        
        plt.title('Base Stats Distribution by Pokemon Type', fontsize=16)
        
        # Add legend and adjust position
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'base_stats_radar_chart_by_type.png'), dpi=300)
        plt.close()
        
        # 3. Create bar charts for each base stat
        for stat in base_stats:
            plt.figure(figsize=(12, 8))
            # Sort by this stat
            sorted_data = type_avg_stats.sort_values(by=stat, ascending=False)
            ax = sorted_data[stat].plot(kind='bar', color='skyblue')
            
            std_stat_name = next((k for k, v in stat_mapping.items() if v == stat), stat)
            plt.title(f'Average {std_stat_name} by Pokemon Type', fontsize=16)
            plt.xlabel('Type', fontsize=14)
            plt.ylabel(f'Average {std_stat_name}', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(sorted_data[stat]):
                ax.text(i, v + 2, f'{v:.1f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'average_{std_stat_name}_by_type.png'), dpi=300)
            plt.close()
    
    # 4. Box plot: Show distribution range of stats for each type
    if total_col:  # If total stats column exists
        plt.figure(figsize=(14, 10))
        
        # Create long-format DataFrame for seaborn
        melted_df = pd.melt(all_pokemon, id_vars=['type'], value_vars=[total_col], 
                           var_name='Stat', value_name='Value')
        
        # Draw box plot
        sns.boxplot(x='type', y='Value', data=melted_df, palette='Set3', hue='type', legend=False)
        plt.title('Total Stats Distribution Range by Pokemon Type', fontsize=16)
        plt.xlabel('Type', fontsize=14)
        plt.ylabel('Total Stats', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_stats_distribution_boxplot.png'), dpi=300)
        plt.close()
    
    # 5. Heat map: Show correlation between different types and stats
    plt.figure(figsize=(16, 12))
    
    # Create new DataFrame with types as rows and stats as columns
    pivot_table = type_avg_stats.copy()
    
    # Draw heat map
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    plt.title('Average Stats Heatmap by Pokemon Type', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stats_heatmap_by_type.png'), dpi=300)
    plt.close()
    
    # 6. Count of Pokemon by type
    plt.figure(figsize=(12, 8))
    type_counts = all_pokemon['type'].value_counts().sort_values(ascending=False)
    
    ax = type_counts.plot(kind='bar', color='lightgreen')
    plt.title('Number of Pokemon by Type', fontsize=16)
    plt.xlabel('Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(type_counts):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pokemon_count_by_type.png'), dpi=300)
    plt.close()
    
    print(f"\nAnalysis complete! All results saved to: {output_dir}")
    
    # Return summary information for report
    summary = {
        'total_pokemon': len(all_pokemon),
        'types': types,  # Now types is defined
        'type_counts': type_counts.to_dict(),
        'avg_stats': type_avg_stats.to_dict(),
    }
    
    return summary

def generate_report(summary):
    """Generate analysis report"""
    if not summary:
        return
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_results')
    
    # Create report file
    report_file = os.path.join(output_dir, 'pokemon_type_analysis_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Pokemon Type Stats Analysis Report\n\n")
        
        f.write("## Data Overview\n\n")
        f.write(f"- Total Pokemon analyzed: {summary['total_pokemon']}\n")
        f.write(f"- Number of Pokemon types: {len(summary['types'])}\n")
        
        # Add Pokemon count by type
        f.write("\n## Number of Pokemon by Type\n\n")
        for type_name, count in sorted(summary['type_counts'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {type_name}: {count}\n")
        
        # Determine types with highest stats
        avg_stats = summary['avg_stats']
        
        # 1. Type with highest total stats
        total_stats_col = next((col for col in avg_stats.keys() if 'total' in col.lower()), None)
        
        if total_stats_col:
            f.write("\n## Stats Analysis\n\n")
            
            # Total stats ranking
            f.write("### Average Total Stats Ranking\n\n")
            
            # Get and sort average total stats for each type
            type_total_stats = {t: avg_stats[total_stats_col][t] for t in summary['types']}
            sorted_by_total = sorted(type_total_stats.items(), key=lambda x: x[1], reverse=True)
            
            for i, (type_name, value) in enumerate(sorted_by_total):
                f.write(f"{i+1}. {type_name}: {value:.1f}\n")
            
            # Strongest type
            strongest_type = sorted_by_total[0][0]
            f.write(f"\nOverall, **{strongest_type}** type Pokemon have the highest average total stats at {sorted_by_total[0][1]:.1f} points.\n")
        
        # 2. Analyze each base stat
        base_stats = [stat for stat in avg_stats.keys() if 'total' not in stat.lower()]
        
        if base_stats:
            f.write("\n### Types with Highest Individual Stats\n\n")
            
            for stat in base_stats:
                # Get and sort this stat value for each type
                type_stat_values = {t: avg_stats[stat][t] for t in summary['types']}
                sorted_by_stat = sorted(type_stat_values.items(), key=lambda x: x[1], reverse=True)
                
                top_type = sorted_by_stat[0][0]
                f.write(f"- **{stat}**: {top_type} type is highest with average {sorted_by_stat[0][1]:.1f}\n")
        
        f.write("\n## Conclusions and Findings\n\n")
        f.write("Based on the analysis of Pokemon stats by type, we can draw the following conclusions:\n\n")
        
        if total_stats_col:
            # Get top 3 and bottom 3 types
            top3 = sorted_by_total[:3]
            bottom3 = sorted_by_total[-3:]
            
            f.write(f"1. **Strongest Types**: {', '.join([t[0] for t in top3])} types have the highest average total stats, ranking in the top three.\n")
            f.write(f"2. **Weakest Types**: {', '.join([t[0] for t in bottom3])} types have lower average total stats, ranking in the bottom three.\n")
        
        if base_stats:
            # Analyze special cases
            stat_champions = {}
            for stat in base_stats:
                type_stat_values = {t: avg_stats[stat][t] for t in summary['types']}
                top_type = max(type_stat_values.items(), key=lambda x: x[1])[0]
                
                if top_type in stat_champions:
                    stat_champions[top_type].append(stat)
                else:
                    stat_champions[top_type] = [stat]
            
            # Find types that excel in multiple stats
            versatile_types = [t for t, stats in stat_champions.items() if len(stats) > 1]
            
            if versatile_types:
                f.write(f"3. **Versatile Types**: ")
                for t in versatile_types:
                    f.write(f"{t} type excels in {', '.join(stat_champions[t])}; ")
                f.write("\n")
        
        f.write("\n4. **Type Characteristics**:\n")
        
        # Here you can add specific observations about type characteristics
        # based on the data analysis results
        
        f.write("\n*This report was automatically generated based on Pokemon type stats analysis*\n")
    
    print(f"Analysis report generated: {report_file}")

def main():
    # Load data
    type_data = load_pokemon_data()
    
    if not type_data:
        print("Could not load data, analysis terminated")
        return
    
    # Analyze data
    summary = analyze_stats_by_type(type_data)
    
    # Generate report
    if summary:  # Only generate report if summary is not None
        generate_report(summary)

if __name__ == "__main__":
    main()
