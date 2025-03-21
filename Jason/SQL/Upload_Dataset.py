import pandas as pd
from sqlalchemy import create_engine, text, TEXT, VARCHAR, BOOLEAN

# 读取CSV文件
df = pd.read_csv('D:\\python\\Code\\DPW_Pokemon\\archive\\pokemon.csv')

# 空值处理（保持原有逻辑）
num_cols = ['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']
df[num_cols] = df[num_cols].fillna(0)
df['percentage_male'] = df['percentage_male'].fillna(-1)
df['type2'] = df['type2'].where(pd.notnull(df['type2']), None)

# 创建数据库连接
engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/pokemon?charset=utf8mb4')

# 创建数据表（使用原生SQL）
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS pokemon (
        pokedex_number INT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        japanese_name VARCHAR(255),
        abilities TEXT,
        type1 VARCHAR(20) NOT NULL,
        type2 VARCHAR(20),
        hp INT DEFAULT 0,
        attack INT DEFAULT 0,
        defense INT DEFAULT 0,
        sp_attack INT DEFAULT 0,
        sp_defense INT DEFAULT 0,
        speed INT DEFAULT 0,
        height_m FLOAT,
        weight_kg FLOAT,
        percentage_male FLOAT DEFAULT -1,
        base_total INT,
        capture_rate INT,
        base_happiness INT,
        base_egg_steps INT,
        experience_growth BIGINT,
        against_bug FLOAT,
        against_dark FLOAT,
        against_dragon FLOAT,
        against_electric FLOAT,
        against_fairy FLOAT,
        against_fight FLOAT,
        against_fire FLOAT,
        against_flying FLOAT,
        against_ghost FLOAT,
        against_grass FLOAT,
        against_ground FLOAT,
        against_ice FLOAT,
        against_normal FLOAT,
        against_poison FLOAT,
        against_psychic FLOAT,
        against_rock FLOAT,
        against_steel FLOAT,
        against_water FLOAT,
        classfication VARCHAR(255),
        generation INT,
        is_legendary BOOLEAN,
        INDEX idx_type1 (type1),
        INDEX idx_type2 (type2)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """))
    conn.commit()

# 数据导入（简化dtype映射）
try:
    df.to_sql(
        name='pokemon',
        con=engine,
        if_exists='append',
        index=False,
        chunksize=500,
        dtype={
            'abilities': TEXT,          # 使用SQLAlchemy的TEXT类型
            'type2': VARCHAR(20),       # 使用VARCHAR类型并指定长度
            'is_legendary': BOOLEAN     # 使用SQLAlchemy的BOOLEAN类型
        }
    )
    print(f"成功导入 {len(df)} 条记录")
except Exception as e:
    print("导入错误:", str(e))