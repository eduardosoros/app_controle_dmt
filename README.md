
# App Controle DMT - Streamlit + SQL Server + CSV

Este projeto é uma solução para análise de DMT por setor, com um fluxo automatizado:

✅ Consulta ao SQL Server local  
✅ Geração automática de CSV  
✅ Push automático para GitHub  
✅ Deploy automático no Streamlit Cloud

## Estrutura de Pastas

```
app_controle_dmt/
├── app_dmt.py                # App Streamlit
├── atualiza_dmt.py           # Script de atualização automática
├── requirements.txt          # Dependências
└── dados/
    └── dmt.csv               # CSV gerado automaticamente
```

## Como atualizar os dados:

1. O script `atualiza_dmt.py` executa a query no SQL Server.
2. Gera o `dados/dmt.csv`.
3. Faz commit e push automático para o GitHub.
4. O Streamlit Cloud detecta e redeploya o app automaticamente.

## Rodando localmente:

```bash
pip install -r requirements.txt
streamlit run app_dmt.py
```

## Observação:

- O app no Streamlit Cloud **sempre lerá o CSV mais recente presente no repositório GitHub**.
