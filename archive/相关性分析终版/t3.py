import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（自动处理编码问题）
try:
    df = pd.read_csv('pokemon_filtered.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('pokemon_filtered.csv', encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv('pokemon_filtered.csv', encoding='latin1')

# 移除不需要的列
df = df.drop(columns=['name'], errors='ignore')

# 计算相关系数
correlations = {}
for column in df.columns:
    if column != 'total_stats' and pd.api.types.is_numeric_dtype(df[column]):
        corr, _ = pearsonr(df[column], df['total_stats'])
        correlations[column] = corr

# 按绝对值排序
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# --- 可视化改进 ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=[corr for _, corr in sorted_correlations],
    y=[attr for attr, _ in sorted_correlations],
    palette="viridis"  # 使用渐变色
)

# 添加相关系数标签
for i, (attr, corr) in enumerate(sorted_correlations):
    ax.text(
        x=corr + 0.02 if corr >=0 else corr - 0.05,  # 正数标签右对齐，负数左对齐
        y=i,
        s=f"{corr:.3f}",  # 显示3位小数
        va='center',
        ha='left' if corr <0 else 'left',
        fontsize=10,
        color='black'
    )

# 图表装饰
plt.title("各属性与Total Stats的Pearson相关系数", pad=20)
plt.xlabel("Pearson相关系数", labelpad=10)
plt.ylabel("属性", labelpad=10)
plt.xlim(-0.15, 0.55)  # 扩展x轴范围避免标签被截断
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # 添加零参考线

plt.tight_layout()
plt.savefig('correlation_annotated.png', dpi=300, bbox_inches='tight')
print("已生成带标注的图表: correlation_annotated.png")