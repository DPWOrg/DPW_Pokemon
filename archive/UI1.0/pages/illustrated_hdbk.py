import streamlit as st
import pandas as pd

st.set_page_config(page_title="宝可梦图鉴")

df = pd.read_csv("data/pokemon.csv", encoding='latin1')
if 'name' not in df.columns:
    df['name'] = ['宝可梦' + str(i + 1) for i in range(len(df))]

st.title("📘 宝可梦图鉴")

search_name = st.text_input("🔍 输入宝可梦名称")
if search_name:
    df = df[df['name'].str.contains(search_name, case=False)]

st.dataframe(df[[
    "pokedex_number", "name", "type1", "type2", "sp_attack",
    "sp_defense", "speed", "weight_kg", "generation", "is_legendary"
]])
