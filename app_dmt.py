#!/usr/bin/env python
# coding: utf-8







import pyodbc
import pandas as pd

# Parâmetros da conexão
server = '192.168.5.224'
database = 'mining_control_mvv'
username = 'userBI'
password = 'pE^437i&z@7P'
port = '1433'

# String de conexão
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server},{port};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)

# Estabelecer conexão
conn = pyodbc.connect(conn_str)

# Consulta SQL (pode ser SELECT * ou um filtro)
query = """
SELECT
    data_hora_fim,
    data_fim,
    turno,
    caminhao,
    origem_subarea,
    destino_subarea,
    material,
    grupo_material,
    tipo_movimentacao,
    balanca_manager,
    dist_cheio AS DMT_Cheio,
    tipo_ciclo
FROM vw_movimentacao_detalhada
WHERE
    YEAR(data_hora_fim) >= 2025
    AND MONTH(data_hora_fim) >= 5
    AND origem_subarea LIKE 'SETOR%'
    AND destino_subarea = 'BRITADOR'
    AND tipo_ciclo <> 'Edit_Delete'
"""

# Carregar no pandas
df = pd.read_sql(query, conn)

# Fechar conexão
conn.close()

# Exibir as primeiras linhas
df.head()




df['tipo_ciclo'].unique()


# ### Percentual de valores nulos por colunas



# Percentual de valores nulos por coluna
percentual_nulos = df.isnull().mean() * 100

# Organiza em forma de tabela ordenada
tabela_nulos = percentual_nulos.reset_index()
tabela_nulos.columns = ['Coluna', 'Percentual_NaN']
tabela_nulos = tabela_nulos.sort_values(by='Percentual_NaN', ascending=False)

# Exibir
print("Percentual de valores nulos por coluna:")
display(tabela_nulos)


# ### Excluir as linhas com valores NAN



df = df.dropna()




df['origem_subarea'].unique()


# ### Estatísticas da variável DMT_Cheio



# Estatísticas descritivas básicas
mad = (df['DMT_Cheio'] - df['DMT_Cheio'].mean()).abs().mean()
estatisticas = df['DMT_Cheio'].describe()
print("Estatísticas Descritivas:")
print(estatisticas)

# Estatísticas adicionais
print("\nEstatísticas Adicionais:")
print(f"Mediana: {df['DMT_Cheio'].median():.2f}")
print(f"Desvio Absoluto Mediano (MAD): {mad:.2f}")
print(f"Variância: {df['DMT_Cheio'].var():.2f}")
print(f"Desvio padrão: {df['DMT_Cheio'].std():.2f}")
print(f"Coeficiente de Variação (CV): {(df['DMT_Cheio'].std() / df['DMT_Cheio'].mean() * 100):.2f}%")
print(f"Valor mínimo: {df['DMT_Cheio'].min():.2f}")
print(f"Valor máximo: {df['DMT_Cheio'].max():.2f}")
print(f"Nº de valores nulos: {df['DMT_Cheio'].isna().sum()}")


# ### SETOR M



# Filtro por origem_subarea
df_filtrado = df[df['origem_subarea'] == 'SETOR M'].copy()
mad = (df_filtrado['DMT_Cheio'] - df_filtrado['DMT_Cheio'].mean()).abs().mean()

# Estatísticas descritivas básicas
estatisticas = df_filtrado['DMT_Cheio'].describe()
print("Estatísticas Descritivas:")
print(estatisticas)

# Estatísticas adicionais
print("\nEstatísticas Adicionais:")
print(f"Mediana: {df_filtrado['DMT_Cheio'].median():.2f}")
print(f"Desvio Absoluto Mediano (MAD): {mad:.2f}")
print(f"Variância: {df_filtrado['DMT_Cheio'].var():.2f}")
print(f"Desvio padrão: {df_filtrado['DMT_Cheio'].std():.2f}")
print(f"Coeficiente de Variação (CV): {(df_filtrado['DMT_Cheio'].std() / df_filtrado['DMT_Cheio'].mean() * 100):.2f}%")
print(f"Valor mínimo: {df_filtrado['DMT_Cheio'].min():.2f}")
print(f"Valor máximo: {df_filtrado['DMT_Cheio'].max():.2f}")
print(f"Nº de valores nulos: {df_filtrado['DMT_Cheio'].isna().sum()}")


# ### SETOR A



# Filtro por origem_subarea
df_filtrado = df[df['origem_subarea'] == 'SETOR A'].copy()
mad = (df_filtrado['DMT_Cheio'] - df_filtrado['DMT_Cheio'].mean()).abs().mean()

# Estatísticas descritivas básicas
estatisticas = df_filtrado['DMT_Cheio'].describe()
print("Estatísticas Descritivas:")
print(estatisticas)

# Estatísticas adicionais
print("\nEstatísticas Adicionais:")
print(f"Mediana: {df_filtrado['DMT_Cheio'].median():.2f}")
print(f"Desvio Absoluto Mediano (MAD): {mad:.2f}")
print(f"Variância: {df_filtrado['DMT_Cheio'].var():.2f}")
print(f"Desvio padrão: {df_filtrado['DMT_Cheio'].std():.2f}")
print(f"Coeficiente de Variação (CV): {(df_filtrado['DMT_Cheio'].std() / df_filtrado['DMT_Cheio'].mean() * 100):.2f}%")
print(f"Valor mínimo: {df_filtrado['DMT_Cheio'].min():.2f}")
print(f"Valor máximo: {df_filtrado['DMT_Cheio'].max():.2f}")
print(f"Nº de valores nulos: {df_filtrado['DMT_Cheio'].isna().sum()}")


# ### SETOR D



# Filtro por origem_subarea
df_filtrado = df[df['origem_subarea'] == 'SETOR D '].copy()
mad = (df_filtrado['DMT_Cheio'] - df_filtrado['DMT_Cheio'].mean()).abs().mean()

# Estatísticas descritivas básicas
estatisticas = df_filtrado['DMT_Cheio'].describe()
print("Estatísticas Descritivas:")
print(estatisticas)

# Estatísticas adicionais
print("\nEstatísticas Adicionais:")
print(f"Mediana: {df_filtrado['DMT_Cheio'].median():.2f}")
print(f"Desvio Absoluto Mediano (MAD): {mad:.2f}")
print(f"Variância: {df_filtrado['DMT_Cheio'].var():.2f}")
print(f"Desvio padrão: {df_filtrado['DMT_Cheio'].std():.2f}")
print(f"Coeficiente de Variação (CV): {(df_filtrado['DMT_Cheio'].std() / df_filtrado['DMT_Cheio'].mean() * 100):.2f}%")
print(f"Valor mínimo: {df_filtrado['DMT_Cheio'].min():.2f}")
print(f"Valor máximo: {df_filtrado['DMT_Cheio'].max():.2f}")
print(f"Nº de valores nulos: {df_filtrado['DMT_Cheio'].isna().sum()}")


# ### SETOR V



# Filtro por origem_subarea
df_filtrado = df[df['origem_subarea'] == 'SETOR V'].copy()
mad = (df_filtrado['DMT_Cheio'] - df_filtrado['DMT_Cheio'].mean()).abs().mean()

# Estatísticas descritivas básicas
estatisticas = df_filtrado['DMT_Cheio'].describe()
print("Estatísticas Descritivas:")
print(estatisticas)

# Estatísticas adicionais
print("\nEstatísticas Adicionais:")
print(f"Mediana: {df_filtrado['DMT_Cheio'].median():.2f}")
print(f"Desvio Absoluto Mediano (MAD): {mad:.2f}")
print(f"Variância: {df_filtrado['DMT_Cheio'].var():.2f}")
print(f"Desvio padrão: {df_filtrado['DMT_Cheio'].std():.2f}")
print(f"Coeficiente de Variação (CV): {(df_filtrado['DMT_Cheio'].std() / df_filtrado['DMT_Cheio'].mean() * 100):.2f}%")
print(f"Valor mínimo: {df_filtrado['DMT_Cheio'].min():.2f}")
print(f"Valor máximo: {df_filtrado['DMT_Cheio'].max():.2f}")
print(f"Nº de valores nulos: {df_filtrado['DMT_Cheio'].isna().sum()}")




df.columns.tolist()


# ### Histogramas por Origem, frequência de DMT



import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Estilo sem grade
sns.set(style="white")

# Lista de subáreas únicas (removendo nulos)
grupos = df['origem_subarea'].dropna().unique()
n = len(grupos)

# Definir layout da grade
ncols = 4
nrows = (n + ncols - 1) // ncols
plt.figure(figsize=(ncols * 5.5, nrows * 4))

# Loop por grupo
for i, grupo in enumerate(sorted(grupos)):
    dados = df[df['origem_subarea'] == grupo]['DMT_Cheio'].dropna()
    if len(dados) < 2:
        continue  # Pula se não houver dados suficientes

    media = np.mean(dados)
    mediana = np.median(dados)

    # Intervalo de confiança da média
    conf_int = stats.t.interval(
        confidence=0.95,
        df=len(dados)-1,
        loc=media,
        scale=stats.sem(dados)
    )

    # Subplot
    plt.subplot(nrows, ncols, i + 1)
    sns.histplot(dados, bins=30, kde=True, color='cornflowerblue', edgecolor='black')
    plt.axvline(media, color='red', linestyle='--', label=f'Média = {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.2f}')
    plt.title(f'DMT_Cheio\n{grupo}', fontsize=9)
    plt.xlabel('DMT_Cheio')
    plt.ylabel('Frequência')

    # IC95%
    plt.text(0.02, 0.95,
             f'IC95%: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]',
             transform=plt.gca().transAxes,
             fontsize=8, verticalalignment='top')

plt.tight_layout()
plt.savefig("histogramas_dmt_por_origem_subarea.png", dpi=300, bbox_inches='tight')
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Estilo sem grade
sns.set(style="white")

# Lista de subáreas únicas (removendo nulos)
grupos = df['origem_subarea'].dropna().unique()
n = len(grupos)

# Layout com 1 gráfico por linha
ncols = 1
nrows = n
plt.figure(figsize=(8, nrows * 4))  # largura 8, altura 4 por gráfico

# Loop por grupo
for i, grupo in enumerate(sorted(grupos)):
    dados = df[df['origem_subarea'] == grupo]['DMT_Cheio'].dropna()
    if len(dados) < 2:
        continue

    media = np.mean(dados)
    mediana = np.median(dados)

    # IC 95%
    conf_int = stats.t.interval(
        confidence=0.95,
        df=len(dados) - 1,
        loc=media,
        scale=stats.sem(dados)
    )

    # Subplot
    plt.subplot(nrows, ncols, i + 1)
    sns.histplot(dados, bins=30, kde=True, color='cornflowerblue', edgecolor='black')
    plt.axvline(media, color='red', linestyle='--', label=f'Média = {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.2f}')
    plt.title(f'Distribuição de DMT_Cheio - {grupo}', fontsize=10)
    plt.xlabel('DMT_Cheio')
    plt.ylabel('Frequência')

    # IC95%
    plt.text(0.02, 0.95,
             f'IC95%: [{conf_int[0]:.2f}, {conf_int[1]:.2f}]',
             transform=plt.gca().transAxes,
             fontsize=8, verticalalignment='top')

plt.tight_layout()
plt.savefig("histogramas_dmt_por_origem_subarea_linha_unica.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:







import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Estilo
sns.set(style="whitegrid")

# Lista de subáreas únicas (sem nulos)
grupos = df['origem_subarea'].dropna().unique()
n = len(grupos)

# Tamanho da figura
ncols = 1
nrows = n
plt.figure(figsize=(10, nrows * 3.5))

# Loop para gerar um boxplot por grupo
for i, grupo in enumerate(sorted(grupos)):
    dados = df[df['origem_subarea'] == grupo]['DMT_Cheio'].dropna()
    if len(dados) < 2:
        continue

    plt.subplot(nrows, ncols, i + 1)
    sns.boxplot(x=dados, color='cornflowerblue', orient='h')
    plt.title(f'Boxplot de DMT_Cheio - {grupo}', fontsize=10)
    plt.xlabel('DMT_Cheio')

plt.tight_layout()
plt.savefig("boxplots_dmt_por_origem_subarea.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:







import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Estilo
sns.set(style="whitegrid")

# Lista de grupos únicos
grupos = df['origem_subarea'].dropna().unique()
n = len(grupos)

# Layout: 1 boxplot por linha
ncols = 1
nrows = n
plt.figure(figsize=(10, nrows * 3.5))

for i, grupo in enumerate(sorted(grupos)):
    dados = df[df['origem_subarea'] == grupo]['DMT_Cheio'].dropna()
    if len(dados) < 2:
        continue

    media = np.mean(dados)
    mediana = np.median(dados)

    # IQR para análise de outliers
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = dados[(dados < lim_inf) | (dados > lim_sup)]
    pct_outliers = 100 * len(outliers) / len(dados)

    # Subplot
    ax = plt.subplot(nrows, ncols, i + 1)
    sns.boxplot(x=dados, color='skyblue', ax=ax)
    plt.axvline(media, color='red', linestyle='--', label=f'Média = {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.2f}')
    plt.title(f'Boxplot - DMT_Cheio | {grupo}', fontsize=10)
    plt.xlabel('DMT_Cheio')

    # Texto de outliers
    plt.text(0.02, 0.9,
             f'Outliers: {len(outliers)} ({pct_outliers:.1f}%)\nLimites: [{lim_inf:.2f}, {lim_sup:.2f}]',
             transform=ax.transAxes,
             fontsize=8, color='darkred', verticalalignment='top')

    plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("boxplots_dmt_por_origem_subarea.png", dpi=300, bbox_inches='tight')
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Estilo
sns.set(style="whitegrid")

# Grupos únicos
grupos = df['origem_subarea'].dropna().unique()
n = len(grupos)

# Layout: 1 boxplot por linha
ncols = 1
nrows = n
plt.figure(figsize=(10, nrows * 4.5))

for i, grupo in enumerate(sorted(grupos)):
    dados = df[df['origem_subarea'] == grupo]['DMT_Cheio'].dropna()
    if len(dados) < 2:
        continue

    # Estatísticas
    media = np.mean(dados)
    mediana = np.median(dados)
    minimo = np.min(dados)
    maximo = np.max(dados)

    # IQR para outliers
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers_baixo = dados[dados < lim_inf]
    outliers_cima = dados[dados > lim_sup]

    # Subplot
    ax = plt.subplot(nrows, ncols, i + 1)
    sns.boxplot(x=dados, color='skyblue', ax=ax)
    plt.axvline(media, color='red', linestyle='--', label=f'Média = {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana = {mediana:.2f}')
    plt.title(f'Boxplot - DMT_Cheio | {grupo}', fontsize=10)
    plt.xlabel('DMT_Cheio')

    # Análise textual
    analise = (
        f"Mediana: {mediana:.2f} | Mínimo: {minimo:.2f} | Máximo: {maximo:.2f}\n"
        f"Outliers abaixo: {len(outliers_baixo)} | acima: {len(outliers_cima)}"
    )
    plt.text(0.02, -0.4, analise,
             transform=ax.transAxes,
             fontsize=9, color='black', va='top')

    plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("boxplots_dmt_com_analise_por_origem_subarea.png", dpi=300, bbox_inches='tight')
plt.show()


# ### SETOR V, quantos pontos acima da média



# Filtrar apenas a subárea SETOR V
df_v = df[df['origem_subarea'] == 'SETOR V'].copy()

# Calcular a média
media_v = df_v['DMT_Cheio'].mean()

# Contar quantos valores estão acima da média
acima_media = (df_v['DMT_Cheio'] > 3000).sum()

# Exibir resultados
print(f"Média de DMT_Cheio em SETOR V: {media_v:.2f}")
print(f"Número de pontos acima da média: {acima_media}")


# In[ ]:




