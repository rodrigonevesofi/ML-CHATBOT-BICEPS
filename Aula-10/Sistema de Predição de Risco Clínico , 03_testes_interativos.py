"""
=============================================================
PROJETO: Sistema de Predição de Risco Clínico
ARQUIVO: 03_testes_interativos.py
DESCRIÇÃO: Retreina o modelo com pacientes.csv e executa
           3 testes consecutivos inserindo dados em tela.
=============================================================
"""

import numpy  as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier

# ─────────────────────────────────────────────
# BLOCO 1 – TREINAR O MODELO (com pacientes.csv)
# ─────────────────────────────────────────────
print("\n" + "═"*58)
print("  CARREGANDO E TREINANDO MODELO COM pacientes.csv")
print("═"*58)

df = pd.read_csv("pacientes.csv")

FEATURES = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
X = df[FEATURES]
y = df["risco"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm  = scaler.transform(X_teste)

modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_treino_norm, y_treino)

acuracia = modelo.score(X_teste_norm, y_teste)
print(f"✅ Modelo treinado! Acurácia no teste: {acuracia:.2%}")
print(f"   Dataset: {len(df)} pacientes | Treino: {len(X_treino)} | Teste: {len(X_teste)}")

# ─────────────────────────────────────────────
# BLOCO 2 – FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────────
ROTULOS  = {0: "BAIXO RISCO 🟢", 1: "RISCO MÉDIO 🟡", 2: "RISCO ALTO 🔴"}
ICONES   = {0: "🟢", 1: "🟡", 2: "🔴"}
RECOMEND = {
    0: "Manter hábitos saudáveis. Retorno anual.",
    1: "Investigar fatores alterados. Consulta em 6 meses.",
    2: "⚠️  Encaminhamento IMEDIATO a especialista.",
}

def prever_paciente(nome, idade, glicose, pressao, imc, colesterol, numero_teste):
    """Normaliza os dados e retorna a predição com probabilidades."""
    dados = pd.DataFrame([{
        "idade": idade, "glicose": glicose,
        "pressao_arterial": pressao, "imc": imc, "colesterol": colesterol
    }])
    dados_norm   = scaler.transform(dados)
    classe       = modelo.predict(dados_norm)[0]
    probs        = modelo.predict_proba(dados_norm)[0]

    print("\n" + "╔" + "═"*54 + "╗")
    print(f"║  TESTE {numero_teste} – RESULTADO DA PREDIÇÃO" + " "*25 + "║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Paciente      : {nome:<37}║")
    print(f"║  Idade         : {str(idade) + ' anos':<37}║")
    print(f"║  Glicose       : {str(glicose) + ' mg/dL':<37}║")
    print(f"║  Pressão Art.  : {str(pressao) + ' mmHg':<37}║")
    print(f"║  IMC           : {str(imc) + ' kg/m²':<37}║")
    print(f"║  Colesterol    : {str(colesterol) + ' mg/dL':<37}║")
    print("╠" + "═"*54 + "╣")
    print(f"║  DIAGNÓSTICO   : {ROTULOS[classe]:<36}║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Probabilidade Baixo Risco : {probs[0]*100:>5.1f}%              ║")
    print(f"║  Probabilidade Risco Médio : {probs[1]*100:>5.1f}%              ║")
    print(f"║  Probabilidade Risco Alto  : {probs[2]*100:>5.1f}%              ║")
    print("╠" + "═"*54 + "╣")
    rec = RECOMEND[classe]
    # quebra a recomendação em linha de até 52 chars
    while len(rec) > 52:
        corte = rec[:52].rfind(" ")
        print(f"║  {rec[:corte]:<52}║")
        rec = rec[corte+1:]
    print(f"║  {rec:<52}║")
    print("╚" + "═"*54 + "╝")

# ─────────────────────────────────────────────
# BLOCO 3 – FUNÇÃO DE ENTRADA EM TELA
# ─────────────────────────────────────────────
def ler_dados_paciente(numero_teste):
    """Lê os dados do paciente via input() em tela."""
    print("\n" + "─"*58)
    print(f"  📋 TESTE {numero_teste} – INSERÇÃO DE DADOS DO PACIENTE")
    print("─"*58)
    print("  Informe os dados clínicos do paciente:\n")

    nome      = input("  Nome do paciente          : ").strip()
    idade     = int(input("  Idade (18–99 anos)        : "))
    glicose   = float(input("  Glicose (mg/dL)           : "))
    pressao   = float(input("  Pressão arterial (mmHg)   : "))
    imc       = float(input("  IMC (kg/m²)               : "))
    colesterol= float(input("  Colesterol total (mg/dL)  : "))

    return nome, idade, glicose, pressao, imc, colesterol

# ─────────────────────────────────────────────
# BLOCO 4 – 3 TESTES CONSECUTIVOS
# ─────────────────────────────────────────────
print("\n" + "═"*58)
print("  INICIANDO 3 TESTES COM INSERÇÃO DE DADOS EM TELA")
print("═"*58)

for i in range(1, 4):
    nome, idade, glicose, pressao, imc, colesterol = ler_dados_paciente(i)
    prever_paciente(nome, idade, glicose, pressao, imc, colesterol, i)
    if i < 3:
        input("\n  ↩  Pressione ENTER para o próximo paciente...")

print("\n" + "═"*58)
print("  3 TESTES CONCLUÍDOS COM SUCESSO!")
print("═"*58 + "\n")
