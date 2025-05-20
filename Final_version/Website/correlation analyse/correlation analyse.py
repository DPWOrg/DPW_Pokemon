import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Read data 
try:
    df = pd.read_csv('pokemon_filtered.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('pokemon_filtered.csv', encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('pokemon_filtered.csv', encoding='latin1')

# Remove unnecessary columns
df = df.drop(columns=['name'], errors='ignore')

# Calculate correlation coefficients
correlations = {}
for column in df.columns:
    if column != 'total_stats' and pd.api.types.is_numeric_dtype(df[column]):
        corr, _ = pearsonr(df[column], df['total_stats'])
        correlations[column] = corr

# Sort by absolute value
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# Visualization improvements 
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=[corr for _, corr in sorted_correlations],
    y=[attr for attr, _ in sorted_correlations],
    palette="viridis"  # Use gradient colors
)

# Add correlation coefficient labels
for i, (attr, corr) in enumerate(sorted_correlations):
    ax.text(
        x=corr + 0.02 if corr >=0 else corr - 0.05,  # Right align for positive numbers, left align for negative
        y=i,
        s=f"{corr:.3f}",  # Display 3 decimal places
        va='center',
        ha='left' if corr <0 else 'left',
        fontsize=10,
        color='black'
    )

# Chart decoration
plt.title("Pearson Correlation Coefficients of Attributes with Total Stats", pad=20)
plt.xlabel("Pearson Correlation Coefficient", labelpad=10)
plt.ylabel("Attribute", labelpad=10)
plt.xlim(-0.15, 0.55)  # Extend x-axis range to avoid label truncation
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Add zero reference line

plt.tight_layout()
plt.savefig('correlation_annotated.png', dpi=300, bbox_inches='tight')
print("Annotated chart generated: correlation_annotated.png")
