"""
=============================================================
GERADOR DE DATASET SINTÉTICO BIOMÉDICO - pacientes.csv
Sistema de Predição de Risco Clínico
=============================================================
Variáveis: nome, idade, glicose, pressão arterial, IMC, colesterol
Alvo: risco (0=baixo, 1=médio, 2=alto)
=============================================================
"""

import numpy as np
import pandas as pd

# ── Reprodutibilidade ─────────────────────────────────────
np.random.seed(42)
N = 2000

# ── Pool de nomes fictícios simples (sem sobrenome) ───────
NOMES = [
    "Ana", "Bruno", "Carla", "Diego", "Elena", "Fábio", "Gisele", "Hugo",
    "Íris", "Jonas", "Karen", "Lucas", "Marta", "Neto", "Olga", "Paulo",
    "Quésia", "Rafael", "Sara", "Tiago", "Ursula", "Vitor", "Wanda", "Xande",
    "Yara", "Zeca", "Aline", "Beto", "Cíntia", "Davi", "Estela", "Felipe",
    "Graça", "Hélio", "Ingrid", "Jorge", "Lúcia", "Marcos", "Nina", "Oscar",
    "Patrícia", "Quirino", "Renata", "Sérgio", "Tânia", "Ulisses", "Vera",
    "Wagner", "Ximena", "Zilda", "Adão", "Beatriz", "César", "Dalva", "Éder",
    "Flávia", "Gilson", "Helena", "Ivan", "Joana", "Kléber", "Larissa",
    "Milton", "Nadir", "Odete", "Pedro", "Quitéria", "Rodrigo", "Simone",
    "Tadeu", "Umberto", "Valéria", "Wendell", "Xerxes", "Yasmin", "Zuleica"
]

nomes = np.random.choice(NOMES, size=N, replace=True)

# ── Geração de variáveis com distribuições realistas ─────

# Idade: 18–99, levemente enviesada para adultos de meia-idade
idade = np.clip(
    np.random.normal(loc=45, scale=18, size=N).astype(int),
    18, 99
)

# Glicose (mg/dL): normal ~70–99, pré-diabético 100–125, diabético ≥126
glicose = np.clip(
    np.random.lognormal(mean=4.7, sigma=0.18, size=N).astype(int),
    60, 400
)

# Pressão arterial sistólica (mmHg): normal ~110–130
pressao_arterial = np.clip(
    np.random.normal(loc=120, scale=18, size=N).astype(int),
    80, 220
)

# IMC (kg/m²): normal 18.5–24.9; distribuição bimodal simulada
imc_base = np.random.normal(loc=26.5, scale=5.5, size=N)
imc = np.clip(np.round(imc_base, 1), 14.0, 55.0)

# Colesterol total (mg/dL): desejável <200, borderline 200–239, alto ≥240
colesterol = np.clip(
    np.random.normal(loc=210, scale=40, size=N).astype(int),
    100, 400
)

# ── Regra de risco clínico coerente ──────────────────────
# Cada variável gera pontos de risco individuais; a soma decide a classe.

def pontos_glicose(g):
    return np.where(g < 100, 0, np.where(g < 126, 1, 2))

def pontos_pressao(p):
    return np.where(p < 130, 0, np.where(p < 160, 1, 2))

def pontos_imc(i):
    return np.where(i < 25, 0, np.where(i < 30, 1, 2))

def pontos_colesterol(c):
    return np.where(c < 200, 0, np.where(c < 240, 1, 2))

def pontos_idade(a):
    return np.where(a < 35, 0, np.where(a < 55, 1, 2))

score = (
    pontos_glicose(glicose) +
    pontos_pressao(pressao_arterial) +
    pontos_imc(imc) +
    pontos_colesterol(colesterol) +
    pontos_idade(idade)
)
# Score varia de 0 a 10
# Adiciona ruído leve (±1 ponto) para tornar os dados menos perfeitamente lineares
ruido = np.random.choice([-1, 0, 0, 1], size=N)
score = np.clip(score + ruido, 0, 10)

# Classificação:
#   0–3  → risco baixo  (0)
#   4–6  → risco médio  (1)
#   7–10 → risco alto   (2)
risco = np.where(score <= 3, 0, np.where(score <= 6, 1, 2))

# ── Montagem do DataFrame ────────────────────────────────
df = pd.DataFrame({
    "nome":            nomes,
    "idade":           idade,
    "glicose":         glicose,
    "pressao_arterial": pressao_arterial,
    "imc":             imc,
    "colesterol":      colesterol,
    "risco":           risco
})

# ── Salvamento ───────────────────────────────────────────
output_path = "pacientes.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ── Relatório de validação ────────────────────────────────
print("=" * 55)
print("   DATASET GERADO COM SUCESSO — pacientes.csv")
print("=" * 55)
print(f"\nRegistros totais : {len(df)}")
print(f"Colunas          : {list(df.columns)}")
print(f"\n{'─'*55}")
print("DISTRIBUIÇÃO DA VARIÁVEL ALVO (risco):")
dist = df["risco"].value_counts().sort_index()
labels = {0: "0 = Baixo risco", 1: "1 = Risco médio", 2: "2 = Risco alto"}
for k, v in dist.items():
    print(f"  {labels[k]:<20}: {v:>5} registros ({v/N*100:.1f}%)")

print(f"\n{'─'*55}")
print("ESTATÍSTICAS DESCRITIVAS:")
print(df.describe().round(2).to_string())

print(f"\n{'─'*55}")
print("PRIMEIROS 5 REGISTROS:")
print(df.head().to_string(index=False))
print("=" * 55)
