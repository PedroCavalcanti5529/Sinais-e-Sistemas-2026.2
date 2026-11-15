"""
Parte 3.4 - Comparação de domínios, redundância e seleção de features (ACC, R1)

Entrada (rodar 3.2 e 3.3 antes):
  tamanho_efeito_r1_por_velocidade.csv  (3.2, AUC das features temporais)
  tamanho_efeito_espectral.csv          (3.3, AUC das features espectrais, já em R1)
  features_tempo_longo.csv              (3.2)
  features_espectrais_longo.csv         (3.3)

Saídas:
  comparacao_dominios.csv     AUC simétrica por domínio × feature × velocidade
  figuras_3_4/auc_dominios.png    heatmap da tabela acima (achado 3.4.1 / 3.4.2)
  figuras_3_4/correlacao.png      Spearman entre as features candidatas
  features_removidas.csv      cada feature removida, com quem é redundante e rho
  features_selecionadas.csv   conjunto final para a 3.5

Critério de seleção (reprodutível):
  - DC (media, rms, potencia) fica fora: só diagnóstico de confundimento de sessão.
  - Ordena as features por AUC média entre velocidades (maior primeiro).
  - Percorre a lista: mantém a feature se |rho| < LIMIAR com todas as já mantidas;
    senão remove e registra com quem é redundante.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT_DIR = Path("/home/pedro/ML-Sinais/outputs")
FIG_DIR = OUT_DIR / "figuras_3_4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DC = ["media", "rms", "potencia"]
LIMIAR = 0.85
FEATS_TEMPO = ["std", "pico_abs", "pico_pico", "crista", "assimetria", "curtose"]
FEATS_ESPECTRAIS = ["freq_dominante", "mag_dominante", "centroide", "entropia",
                    "largura_banda", "potencia_B1", "potencia_B2", "potencia_B3"]


def tabela_auc():
    t = pd.read_csv(OUT_DIR / "tamanho_efeito_r1_por_velocidade.csv")
    e = pd.read_csv(OUT_DIR / "tamanho_efeito_espectral.csv")
    t["dominio"], e["dominio"] = "Tempo", "Frequência"
    d = pd.concat([x[x.sensor == "ACC"] for x in (t, e)])
    d = d[d.feature != "media"]                    # média é só diagnóstico
    d = d[["dominio", "feature", "speed", "auc_sep"]]
    d.to_csv(OUT_DIR / "comparacao_dominios.csv", index=False)
    return d


def features_janela():
    t = pd.read_csv(OUT_DIR / "features_tempo_longo.csv")
    e = pd.read_csv(OUT_DIR / "features_espectrais_longo.csv")
    q = lambda d: d[(d.sensor == "ACC") & (d.repetition == 1)]
    t, e = q(t), q(e)
    ch = ["record_id", "window_idx"]
    df = t[ch + FEATS_TEMPO].merge(e[ch + FEATS_ESPECTRAIS], on=ch)
    print(f"Dataset mesclado: {len(df)} janelas de ACC em R1.")
    return df, FEATS_TEMPO + FEATS_ESPECTRAIS


def plot_auc(d):
    d = d.assign(rotulo=d.feature.map(lambda f: f + " (DC)" if f in DC else f))
    piv = d.pivot_table(index=["dominio", "rotulo"], columns="speed",
                        values="auc_sep")
    piv.columns = [f"SP{c}" for c in piv.columns]
    plt.figure(figsize=(5, 7))
    sns.heatmap(piv, annot=True, fmt=".2f", vmin=0.5, vmax=1, cmap="viridis")
    plt.title("AUC simétrica (ACC, R1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "auc_dominios.png", dpi=150)
    plt.close()
    return piv


def selecionar(d, df, feats):
    corr = df[feats].corr(method="spearman")
    plt.figure(figsize=(9, 8))
    sns.heatmap(corr, annot=True, fmt=".1f", annot_kws={"size": 7},
                cmap="coolwarm", center=0)
    plt.title("Correlação de Spearman (ACC, R1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlacao.png", dpi=150)
    plt.close()

    score = (d[~d.feature.isin(DC) & d.feature.isin(feats)]
             .groupby("feature").auc_sep.mean())
    mantidas, removidas = [], []
    for f in score.sort_values(ascending=False).index:
        par = next((m for m in mantidas if abs(corr.loc[f, m]) >= LIMIAR), None)
        if par is None:
            mantidas.append(f)
        else:
            removidas.append((f, par, round(corr.loc[f, par], 2),
                              round(score[f], 3), round(score[par], 3)))

    rem = pd.DataFrame(removidas, columns=["removida", "redundante_com", "rho",
                                           "auc_removida", "auc_mantida"])
    rem.to_csv(OUT_DIR / "features_removidas.csv", index=False)
    sel = pd.DataFrame({"features_selecionadas": mantidas,
                        "auc_media": [round(score[f], 3) for f in mantidas]})
    sel.to_csv(OUT_DIR / "features_selecionadas.csv", index=False)
    return sel, rem


if __name__ == "__main__":
    d = tabela_auc()
    piv = plot_auc(d)
    print("\n=== AUC simétrica por domínio, feature e velocidade ===")
    print(piv.round(2).to_string())

    df, feats = features_janela()
    sel, rem = selecionar(d, df, feats)
    print(f"\n=== Removidas por redundância (|rho| >= {LIMIAR}) ===")
    print(rem.to_string(index=False) if len(rem) else "nenhuma")
    print("\n=== Conjunto selecionado para a 3.5 ===")
    print(sel.to_string(index=False))