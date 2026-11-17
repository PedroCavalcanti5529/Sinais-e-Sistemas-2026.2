"""
Parte 3.4 - Comparação de domínios, redundância e seleção de features (ACC, R1)  (v2)

Entrada (rodar 3.2 e 3.3 antes):
  tamanho_efeito_r1_por_velocidade.csv  (3.2, AUC das features temporais)
  tamanho_efeito_espectral.csv          (3.3, AUC das features espectrais, já em R1)
  features_tempo_longo.csv              (3.2)
  features_espectrais_longo.csv         (3.3)

Saídas:
  figuras_3_4/auc_dominios.png        heatmap de AUC por domínio × feature × velocidade
  figuras_3_4/correlacao.png          Spearman (média das matrizes DENTRO de cada velocidade)
  features_removidas.csv              feature removida, com quem é redundante e rho (por domínio)
  features_selecionadas.csv           selecionadas, POR DOMÍNIO (Tempo / Frequência)
  conjuntos_modelos.csv               NOVO: modelos A (tempo), B (espectral), C (união)
  redundancia_entre_dominios.csv      NOVO: pares tempo × espectral com |rho| >= limiar
  tabela_features_ml_ACC.csv          NOVO: uma linha por janela (features + classe + y +
                                      record_id + speed/channel/fault/severity) p/ a 3.5

Critério de seleção (reprodutível), aplicado SEPARADAMENTE em cada domínio:
  - DC (media, rms, potencia) fica fora: só diagnóstico de confundimento de sessão.
  - Ordena as features do domínio por AUC média entre velocidades (maior primeiro).
  - Percorre a lista: mantém a feature se |rho| < LIMIAR com todas as já mantidas
    DO MESMO DOMÍNIO; senão remove e registra com quem é redundante.
  - Seleção por domínio (e não global) porque a 3.5 exige o modelo A só com tempo e o B
    só com espectro; uma seleção global poderia tirar uma feature temporal por ser
    redundante com uma espectral e desbalancear A e B. C = A ∪ B.

Cuidados contra vazamento / correlação espúria (mudanças em relação à v1):
  - Correlação calculada dentro de cada velocidade e depois promediada: a velocidade
    escala quase todas as amplitudes e, agrupada, infla a correlação.
  - A seleção usa rótulos (AUC) → para a 3.5, chamar selecionar(df, treino_ids=...) com
    os record_id de TREINO. A execução deste script (todos os registros) é exploratória
    e serve à análise da 3.4.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (OUT_DIR, FEATS_DC as DC, FEATS_TEMPO_AC as FEATS_TEMPO,
                    FEATS_ESPECTRAIS, LIMIAR_CORR as LIMIAR, REP_PAR)
from utils import auc_rank

FIG_DIR = OUT_DIR / "figuras_3_4"
IDS = ["record_id", "window_idx"]
META = ["classe", "fault", "severity", "speed", "channel", "repetition"]


def tabela_auc():
    """AUC simétrica por domínio × feature × velocidade (a partir dos CSVs da 3.2/3.3).
    Não grava CSV: a v1 gerava comparacao_dominios.csv, subconjunto dos dois de origem."""
    t = pd.read_csv(OUT_DIR / "tamanho_efeito_r1_por_velocidade.csv")
    e = pd.read_csv(OUT_DIR / "tamanho_efeito_espectral.csv")
    t["dominio"], e["dominio"] = "Tempo", "Frequência"
    d = pd.concat([x[x.sensor == "ACC"] for x in (t, e)])
    d = d[d.feature != "media"]                    # média é só diagnóstico
    return d[["dominio", "feature", "speed", "auc_sep"]]


def features_janela():
    """Tabela larga: uma linha por janela de ACC em R1 (tempo + espectro + rótulo)."""
    t = pd.read_csv(OUT_DIR / "features_tempo_longo.csv")
    e = pd.read_csv(OUT_DIR / "features_espectrais_longo.csv")
    q = lambda d: d[(d.sensor == "ACC") & (d.repetition == REP_PAR)]
    t, e = q(t), q(e)
    df = t[IDS + META + DC + FEATS_TEMPO].merge(e[IDS + FEATS_ESPECTRAIS], on=IDS)
    df["y"] = (df["classe"] == "defeituoso").astype(int)    # 1 = defeituoso (positiva)
    print(f"Dataset mesclado: {len(df)} janelas de ACC em R{REP_PAR}.")
    return df


def plot_auc(d):
    d = d.assign(rotulo=d.feature.map(lambda f: f + " (DC)" if f in DC else f))
    piv = d.pivot_table(index=["dominio", "rotulo"], columns="speed", values="auc_sep")
    piv.columns = [f"SP{c}" for c in piv.columns]
    plt.figure(figsize=(6, 0.38 * len(piv) + 1.5))
    sns.heatmap(piv, annot=True, fmt=".2f", vmin=0.5, vmax=1, cmap="viridis")
    plt.title("AUC simétrica (ACC, R1)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "auc_dominios.png", dpi=150)
    plt.close()
    return piv


def auc_features(df, feats):
    """AUC simétrica média entre velocidades, calculada SÓ nas linhas de df."""
    por_vel = []
    for _, g in df.groupby("speed"):
        y = g["y"].to_numpy().astype(bool)
        if y.all() or (~y).all():
            continue
        a = auc_rank(g[feats].to_numpy(), y)
        por_vel.append(np.maximum(a, 1 - a))
    return pd.Series(np.mean(por_vel, axis=0), index=feats)


def corr_intra_velocidade(df, feats):
    """Média das matrizes de Spearman calculadas dentro de cada velocidade."""
    mats = [g[feats].corr(method="spearman") for _, g in df.groupby("speed")]
    return sum(mats) / len(mats)


def _guloso(feats, score, corr, limiar):
    mantidas, removidas = [], []
    for f in score[feats].sort_values(ascending=False).index:
        par = next((m for m in mantidas if abs(corr.loc[f, m]) >= limiar), None)
        if par is None:
            mantidas.append(f)
        else:
            removidas.append((f, par, round(corr.loc[f, par], 2),
                              round(score[f], 3), round(score[par], 3)))
    return mantidas, removidas


def selecionar(df, treino_ids=None, limiar=LIMIAR):
    """Seleção por domínio. treino_ids: record_id de TREINO (use na 3.5); None = todos.
    Retorna (sel, rem, corr, cruz)."""
    d = df if treino_ids is None else df[df.record_id.isin(set(treino_ids))]
    dominios = {"Tempo": FEATS_TEMPO, "Frequência": FEATS_ESPECTRAIS}
    # features constantes não têm correlação definida (ex.: potrel_B4 em IMU)
    dominios = {k: [f for f in v if d[f].nunique() > 1] for k, v in dominios.items()}
    todas = sum(dominios.values(), [])

    score = auc_features(d, todas)
    corr = corr_intra_velocidade(d, todas)

    sel, rem = [], []
    for nome, feats in dominios.items():
        m, r = _guloso(feats, score, corr, limiar)
        sel += [dict(dominio=nome, feature=f, auc_media=round(score[f], 3)) for f in m]
        rem += [dict(dominio=nome, removida=a, redundante_com=b, rho=c,
                     auc_removida=e, auc_mantida=g) for a, b, c, e, g in r]
    sel = pd.DataFrame(sel)
    rem = pd.DataFrame(rem, columns=["dominio", "removida", "redundante_com", "rho",
                                     "auc_removida", "auc_mantida"])

    # redundância entre domínios (informativa; não remove nada)
    cruz = [dict(feat_tempo=a, feat_espectral=b, rho=round(corr.loc[a, b], 2))
            for a in dominios["Tempo"] for b in dominios["Frequência"]
            if abs(corr.loc[a, b]) >= limiar]
    cruz = pd.DataFrame(cruz, columns=["feat_tempo", "feat_espectral", "rho"])
    return sel, rem, corr, cruz


def plot_corr(corr, n_tempo):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".1f", annot_kws={"size": 6},
                cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.axhline(n_tempo, color="k", lw=1.5)
    plt.axvline(n_tempo, color="k", lw=1.5)
    plt.title("Correlação de Spearman (média dentro de cada velocidade; ACC, R1)\n"
              "linha preta: separa features temporais (topo) das espectrais")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlacao.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    d = tabela_auc()
    piv = plot_auc(d)
    print("\n=== AUC simétrica por domínio, feature e velocidade ===")
    print(piv.round(2).to_string())

    df = features_janela()
    df.to_csv(OUT_DIR / "tabela_features_ml_ACC.csv", index=False)

    sel, rem, corr, cruz = selecionar(df)
    n_tempo = int((sel.dominio == "Tempo").sum() + (rem.dominio == "Tempo").sum())
    plot_corr(corr, n_tempo)
    sel.to_csv(OUT_DIR / "features_selecionadas.csv", index=False)
    rem.to_csv(OUT_DIR / "features_removidas.csv", index=False)
    cruz.to_csv(OUT_DIR / "redundancia_entre_dominios.csv", index=False)

    A = sel[sel.dominio == "Tempo"].feature.tolist()
    B = sel[sel.dominio == "Frequência"].feature.tolist()
    modelos = pd.DataFrame({"modelo": ["A (tempo)", "B (espectral)", "C (combinado)"],
                            "n_features": [len(A), len(B), len(A + B)],
                            "features": [";".join(A), ";".join(B), ";".join(A + B)]})
    modelos.to_csv(OUT_DIR / "conjuntos_modelos.csv", index=False)

    print(f"\n=== Removidas por redundância (|rho| >= {LIMIAR}, dentro do domínio) ===")
    print(rem.to_string(index=False) if len(rem) else "nenhuma")
    print("\n=== Redundância ENTRE domínios (informativa) ===")
    print(cruz.to_string(index=False) if len(cruz) else "nenhuma")
    print("\n=== Conjuntos para a 3.5 ===")
    print(modelos.to_string(index=False))
    print("\n[AVISO] Seleção exploratória (todos os registros, usa rótulos). Na 3.5 refaça com "
          "selecionar(df, treino_ids=<record_id de treino>).")
