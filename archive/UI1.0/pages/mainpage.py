import streamlit as st
import pandas as pd

st.set_page_config(page_title="主页")

st.title("🏠 历史配队主页")

# 模拟历史记录
if "history" not in st.session_state:
    st.session_state.history = []

st.write("已保存队伍：")
if st.session_state.history:
    for idx, item in enumerate(st.session_state.history):
        st.write(f"队伍 {idx+1}: {item['team']}，评分：{item['score']}")
else:
    st.write("暂无记录")

if st.button("📤 导出记录"):
    df = pd.DataFrame(st.session_state.history)
    st.download_button("下载 CSV", data=df.to_csv(index=False), file_name="team_history.csv", mime="text/csv")
