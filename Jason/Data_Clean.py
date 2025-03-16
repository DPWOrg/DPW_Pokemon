import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 读取CSV文件
file_path = "../archive/pokemon.csv"
df = pd.read_csv(file_path)

# 处理缺失值
# 用相同 type1 类型的中位数填充 height_m 和 weight_kg
df['height_m'] = df.groupby('type1')['height_m'].apply(lambda x: x.fillna(x.median()))
df['weight_kg'] = df.groupby('type1')['weight_kg'].apply(lambda x: x.fillna(x.median()))

# 处理 percentage_male 缺失值，使用各类宝可梦的均值填充
df['percentage_male'] = df.groupby('type1')['percentage_male'].apply(lambda x: x.fillna(x.mean()))

# 处理 type2 缺失值，填充为 'None'
df['type2'] = df['type2'].fillna('None')

# 转换 capture_rate 为数值类型
df['capture_rate'] = pd.to_numeric(df['capture_rate'], errors='coerce')

# 填充 capture_rate 的缺失值
if df['capture_rate'].isnull().sum() > 0:
    df['capture_rate'].fillna(df['capture_rate'].median(), inplace=True)

# 处理异常值，移除身高大于10米或体重大于500kg的记录（可能为输入错误）
df = df[(df['height_m'] <= 10) & (df['weight_kg'] <= 500)]

# 确保所有数据无缺失值
assert df.isnull().sum().sum() == 0, "数据仍然存在缺失值，请检查预处理逻辑"

# 输出处理后的数据基本信息
df.info()
