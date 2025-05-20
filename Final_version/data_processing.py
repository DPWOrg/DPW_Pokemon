import pandas as pd
import os

# Load the data, I'm doing it in a way that's convenient for me
file_path = 'pokemon.csv'
df = pd.read_csv(file_path)
# Calculate the total stats
df['total_stats'] = df['attack'] + df['defense'] + df['hp'] + df['speed'] + df['sp_attack'] + df['sp_defense']
# Keep the columns needed for battle - related numerical values
selected_df = df[[
    'name',
    'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy',
    'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass',
    'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic',
    'against_rock', 'against_steel', 'against_water',
    'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense', 'total_stats'
]].copy().reset_index(drop=True)
# Check for missing values
missing_values = selected_df.isnull().sum()
total_missing = missing_values.sum()
if total_missing == 0:
    print("No missing values")
else:
    print("Number of missing values in each column:")
    print(missing_values)
# Save the battle numerical values file
output_csv = 'battle_skill.csv'
selected_df.to_csv(output_csv, index=False)
# Save the columns that may be related to analyzing the total stats of Pokémon, used to analyze the possible correlation between the total stats of Pokémon and other values
result2_df = df[['name', 'total_stats', 'is_legendary', 'base_happiness', 'experience_growth', 'base_egg_steps', 'capture_rate', 'weight_kg', 'height_m']]
# Check for missing values
missing_values = result2_df.isnull().sum()
total_missing = missing_values.sum()
if total_missing == 0:
    print("No missing values")
else:
    print("Number of missing values in each column:")
    print(missing_values)
    # Delete rows with missing values and reset index
    result2_df = result2_df.dropna().reset_index(drop=True)
# Save the file for analyzing the correlation information of Pokémon total stats
csv_path = 'pokemon_filtered.csv'
result2_df.to_csv(csv_path, index=False)

# Create a folder
folder_path = 'pokemon_types'
os.makedirs(folder_path, exist_ok=True)

# Get all unique values of type1 and type2
types = pd.concat([df['type1'], df['type2']]).dropna().unique()

# Iterate through each type
for pokemon_type in types:
    # Filter out Pokémon whose type1 or type2 is the current type
    type_df = df[(df['type1'] == pokemon_type) | (df['type2'] == pokemon_type)]

    # Extract the required columns
    result_df = type_df[['name', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense', 'total_stats']]

    # Build the file name
    file_name = f'{pokemon_type}.csv'
    file_path = os.path.join(folder_path, file_name)

    # Save as a CSV file
    result_df.to_csv(file_path, index=False)
