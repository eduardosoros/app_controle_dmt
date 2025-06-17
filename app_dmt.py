import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# Carregar os dados
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('dados/dmt.csv')
    df['data_fim'] = pd.to_datetime(df['data_fim'])
    return df

st.title("📊 Controle DMT - Análise por Setor (via CSV)")

df = load_data()

# ---------------------------------
# Filtro de data
# ---------------------------------
min_date = df['data_fim'].min().date()
max_date = df['data_fim'].max().date()
date_range = st.date_input('Período:', [min_date, max_date])

df_filtered = df[
    (df['data_fim'] >= pd.to_datetime(date_range[0])) &
    (df['data_fim'] <= pd.to_datetime(date_range[1]))
]

# ---------------------------------
# Análise por setor
# ---------------------------------
setores = df_filtered['origem_subarea'].unique()

for setor in sorted(setores):
    st.subheader(f"Setor: {setor}")

    dados = df_filtered[df_filtered['origem_subarea'] == setor]['DMT_Cheio']

    if len(dados) < 5:
        st.warning(f"Poucos dados no setor {setor}.")
        continue

    media = dados.mean()
    mediana = dados.median()
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    lim_inf = q1 - 1.5 * (q3 - q1)
    lim_sup = q3 + 1.5 * (q3 - q1)
    outliers_inf = (dados < lim_inf).sum()
    outliers_sup = (dados > lim_sup).sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dados, bins=30, color='cornflowerblue', edgecolor='black')

    for x, color, label in zip([lim_inf, q1, media, mediana, q3, lim_sup],
                               ['gray', 'blue', 'orange', 'red', 'blue', 'brown'],
                               ['Lim. Inf', 'Q1', 'Média', 'Mediana', 'Q3', 'Lim. Sup']):
        ax.axvline(x, color=color, linestyle='--', label=f'{label}: {x:.2f}')

    ax.legend(fontsize=8)
    ax.set_xlabel('DMT')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

    st.markdown(f"""
    **Estatísticas:**
    - Total: {len(dados)}
    - Média: {media:.2f}
    - Mediana: {mediana:.2f}
    - Q1: {q1:.2f} | Q3: {q3:.2f}
    - Limites: {lim_inf:.2f} a {lim_sup:.2f}
    - Outliers < Inf: {outliers_inf} | > Sup: {outliers_sup}
    """)
