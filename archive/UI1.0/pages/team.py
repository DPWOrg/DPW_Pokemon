import streamlit as st
import pandas as pd

df = pd.read_csv("data/pokemon.csv", encoding='latin1')
if 'name' not in df.columns:
    df['name'] = ['宝可梦' + str(i + 1) for i in range(len(df))]

st.title("🤖 智能配队")

selected = st.multiselect("请选择你拥有的宝可梦", df['name'].tolist())

if st.button("✨ 生成推荐配队"):
    if len(selected) < 3:
        st.warning("请选择至少3个宝可梦")
    else:
        team = selected[:3]
        score = 85 + len(team)
        st.success(f"推荐队伍：{' / '.join(team)}，评分：{score}")
