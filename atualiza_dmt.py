import pandas as pd
import pyodbc
import os

conn_str = (
    "DSN=SQLServerDMT_User;"
    "UID=userBI;"
    "PWD=pE^437i&z@7P"
)

conn = pyodbc.connect(conn_str)

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

df = pd.read_sql(query, conn)
conn.close()

# Exporta o CSV
output_csv = r'C:\Projetos\app_controle_dmt\dados\dmt.csv'
df.to_csv(output_csv, index=False)

# Faz o Git Push
os.chdir(r'C:\Projetos\app_controle_dmt')
os.system('git add dados/dmt.csv')
os.system('git commit -m "Atualização automática via DSN de Usuário"')
os.system('git push')

