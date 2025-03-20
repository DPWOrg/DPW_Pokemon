import pandas as pd
import ast
from datetime import datetime
import matplotlib.pyplot as plt

# Initialize report content
report = []


def add_report_section(title, content=None, df=None, image_path=None):
    """Add content to the report"""
    report.append(f"\n## {title}\n")
    if content:
        report.append(content + "\n")
    if df is not None:
        report.append(df.to_markdown() + "\n")
    if image_path:
        report.append(f"![Visualization]({image_path})\n")


# Record runtime information
start_time = datetime.now()
report.append(f"# Pokémon Data Analysis Report\n\n**Generated on**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Data loading ------------------------------------------------
try:
    df = pd.read_csv('../archive/pokemon_cleaned.csv')
    data_source = '../archive/pokemon_cleaned.csv'
except FileNotFoundError:
    df = pd.read_csv('pokemon.csv')
    data_source = 'pokemon.csv'

add_report_section(
    title="Data Loading",
    content=f"Data Source: `{data_source}`\n\nInitial Data Dimensions: {df.shape}"
)

# Column selection ------------------------------------------------
keep_columns = [
    'name', 'pokedex_number', 'type1', 'type2',
    'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
    'abilities', 'height_m', 'weight_kg', 'generation', 'is_legendary'
]
removed_columns = set(df.columns) - set(keep_columns)
df = df[keep_columns]

add_report_section(
    title="Feature Selection",
    content=(
        f"Number of Features Retained: {len(keep_columns)}\n\n"
        f"Removed Features: {', '.join(removed_columns)}"
    )
)


# Data cleaning ------------------------------------------------
def clean_data(df):
    # Process abilities column
    df['abilities'] = df['abilities'].apply(
        lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else x
    )

    # Handle missing values
    missing_before = df.isnull().sum()
    df['type2'] = df['type2'].fillna('None')
    missing_after = df.isnull().sum()

    # Handle outliers
    height_q75 = df['height_m'].quantile(0.75)
    weight_q75 = df['weight_kg'].quantile(0.75)
    df['height_m'] = df['height_m'].clip(upper=height_q75 * 4)
    df['weight_kg'] = df['weight_kg'].clip(upper=weight_q75 * 4)

    # Derived features
    df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2 + 1e-6)
    df['total_stats'] = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].sum(axis=1)

    # Handle duplicates
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    duplicates_after = df.duplicated().sum()

    # Record cleaning process
    add_report_section(
        title="Data Cleaning",
        content=(
            f"**Missing Values Handling**:\n"
            f"- Missing Values Before:\n{missing_before[missing_before > 0].to_markdown()}\n"
            f"- Missing Values After:\n{missing_after[missing_after > 0].to_markdown()}\n\n"
            f"**Outliers Handling**:\n"
            f"- Height Threshold: {height_q75 * 4:.2f} meters\n"
            f"- Weight Threshold: {weight_q75 * 4:.2f} kilograms\n\n"
            f"**Duplicates Handling**:\n"
            f"- Duplicates Before: {duplicates_before}\n"
            f"- Duplicates After: {duplicates_after}"
        )
    )

    return df


df = clean_data(df)

# Generate visualization
plt.figure(figsize=(10, 6))
df['type1'].value_counts().sort_values().plot(kind='barh')
plt.title('Primary Type Distribution')
plt.savefig('type1_distribution.png', bbox_inches='tight')
add_report_section(
    title="Type Distribution Visualization",
    content="Primary Type Distribution:",
    image_path='type1_distribution.png'
)


# Data analysis ------------------------------------------------
def generate_analysis(df):
    # Basic statistics
    stats_df = df[['hp', 'attack', 'defense', 'speed', 'height_m', 'weight_kg', 'bmi', 'total_stats']].describe().round(
        2)

    # Type analysis
    type_analysis = pd.concat([
        df['type1'].value_counts().rename('Primary Type'),
        df['type2'].value_counts().rename('Secondary Type')
    ], axis=1).fillna(0).astype(int)

    # Legendary Pokémon analysis
    legendary_analysis = df.groupby('is_legendary')['total_stats'].agg(['mean', 'std', 'min', 'max']).round(2)

    # Type combination analysis
    type_comb = df.groupby(['type1', 'type2']).size().sort_values(ascending=False).head(5).reset_index(name='Count')

    # Add analysis results to report
    add_report_section(
        title="Numerical Feature Statistics",
        df=stats_df
    )

    add_report_section(
        title="Type Distribution Analysis",
        df=type_analysis
    )

    add_report_section(
        title="Legendary Pokémon Attribute Analysis",
        df=legendary_analysis
    )

    add_report_section(
        title="Top 5 Common Type Combinations",
        df=type_comb
    )


generate_analysis(df)

# Save processed data --------------------------------------------
output_csv = 'pokemon_processed.csv'
df.to_csv(output_csv, index=False)

# Complete report
processing_time = datetime.now() - start_time
report.append(
    f"\n## Processing Complete\n"
    f"- Output File: `{output_csv}`\n"
    f"- Final Data Dimensions: {df.shape}\n"
    f"- Total Processing Time: {processing_time.total_seconds():.1f} seconds"
)

# Generate Markdown report
with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("Processing complete! Analysis report saved as analysis_report.md")