"""
AULA_09 - Análise e Validação do Dataset de Pacientes
=======================================================
Tarefas:
  1. Identificar e corrigir valores nulos/zero em colunas críticas
     (Glicose, Pressão Arterial, IMC) via substituição pela mediana.
  2. Listar os 10 pacientes com maiores níveis de Glicose e Colesterol.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
#  CARREGAMENTO DO DATASET                                            #
# ------------------------------------------------------------------ #
df = pd.read_csv('pacientes.csv')
print(f"Dataset carregado: {len(df)} registros, {df.shape[1]} colunas")
print(df.head(), "\n")

# ================================================================== #
# TAREFA 1 – Identificação e correção de valores inválidos           #
# ================================================================== #

COLUNAS_CRITICAS = ['glicose', 'pressao_arterial', 'imc']

print("=" * 60)
print("TAREFA 1: IDENTIFICAÇÃO E CORREÇÃO DE VALORES INVÁLIDOS")
print("=" * 60)

print("\n--- ANTES DA CORREÇÃO ---")
for col in COLUNAS_CRITICAS:
    zeros = (df[col] == 0).sum()
    nulos = df[col].isnull().sum()
    print(f"  {col}: {zeros} zeros, {nulos} nulos")

df_corrigido = df.copy()

print("\n--- PROCESSANDO CORREÇÕES ---")
for col in COLUNAS_CRITICAS:
    # Calcula a mediana ignorando zeros (tratados como ausentes)
    mediana = df_corrigido[col].replace(0, np.nan).median()

    zeros_antes = (df_corrigido[col] == 0).sum()
    nulos_antes = df_corrigido[col].isnull().sum()

    # Substitui zeros por NaN, depois preenche NaN pela mediana
    df_corrigido[col] = df_corrigido[col].replace(0, np.nan)
    total_substituicoes = df_corrigido[col].isnull().sum()
    df_corrigido[col] = df_corrigido[col].fillna(mediana)

    print(f"\n  [{col}]")
    print(f"    Zeros encontrados  : {zeros_antes}")
    print(f"    Nulos encontrados  : {nulos_antes}")
    print(f"    Total corrigidos   : {total_substituicoes}")
    print(f"    Mediana aplicada   : {mediana:.2f}")

print("\n--- APÓS CORREÇÃO ---")
for col in COLUNAS_CRITICAS:
    zeros = (df_corrigido[col] == 0).sum()
    nulos = df_corrigido[col].isnull().sum()
    print(f"  {col}: {zeros} zeros, {nulos} nulos")

print("\n✓ Dataset consistente — sem valores nulos ou zeros indevidos.\n")

# Exporta o dataset corrigido
df_corrigido.to_csv('pacientes_corrigidos.csv', index=False)
print("✓ Dataset corrigido salvo em: pacientes_corrigidos.csv\n")


# ================================================================== #
# TAREFA 2 – Top 10 Glicose e Top 10 Colesterol                     #
# ================================================================== #

print("=" * 60)
print("TAREFA 2: TOP 10 PACIENTES")
print("=" * 60)

print("\n🔹 TOP 10 PACIENTES COM MAIOR NÍVEL DE GLICOSE:")
top_glicose = (
    df_corrigido[['nome', 'glicose']]
    .sort_values('glicose', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top_glicose.index += 1
print(top_glicose.to_string())

print("\n🔹 TOP 10 PACIENTES COM MAIOR NÍVEL DE COLESTEROL:")
top_colesterol = (
    df_corrigido[['nome', 'colesterol']]
    .sort_values('colesterol', ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top_colesterol.index += 1
print(top_colesterol.to_string())

print("\n✓ Análise concluída com sucesso!")
