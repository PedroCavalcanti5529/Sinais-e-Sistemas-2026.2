"""
Parte 3.2 - Análise e extração de características no domínio do tempo  (v8)

Entrada  : saídas da 3.1 (janelas_{ACC,IMU}.npy + ..._indice.csv, REMOVE_DC=False)
           + registros_metadados.csv (recupera fault/severity/speed/channel/repetition)
Saídas   : features_tempo_longo.csv             (uma linha por janela × eixo)
           tamanho_efeito_r1_por_velocidade.csv (AUC / d de Cohen, só R1)
           tabela_resumo_velocidade_r1.csv      (média/desvio por classe e velocidade, R1)
           Figuras opcionais via PLOT=True.

Definições (janela x[n]; mu = média; xc = x - mu)
  média mu, potência média P = mean(x^2), RMS = sqrt(P), std sigma (ddof=0)
  pico_abs = max|xc|, pico_pico = max x - min x, crista = pico_abs/sigma
  assimetria = E[(xc/sigma)^3], curtose = E[(xc/sigma)^4] (Pearson, normal=3)
Identidade: P = mu^2 + sigma^2  ->  RMS^2 = mu^2 + sigma^2  (verificada no pipeline)

Notas sobre features
  - FEATS_DC (media, rms, potencia) só servem para diagnóstico de confundimento.
    Não usar em ML: em R1 (mesma sessão) a AUC é ~0.52.
  - FEATS_AC são as features de classificação.

Notas de confundimento
  - R2 e R3 são de outra sessão (nov/2024) e só existem para o saudável.
    Agregar todas as repetições faz o classificador aprender sessão, não falha.
    Por isso todas as tabelas de efeito são restritas a repetition == 1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, skew

OUT_DIR = Path("/home/pedro/ML-Sinais/outputs")
FIG_DIR = OUT_DIR / "figuras"
AXES = {"ACC": ["ch"], "IMU": ["x", "y", "z"]}
UNITS = {"ACC": "g (assumido)", "IMU": "m/s²"}

# --- figuras / relatório ---
PLOT = True                 # False: não gera figuras
CANAL_RELATORIO = 1
PAR_SP = 1
ZOOM_S = 0.05

# --- features ---
# De forma/amplitude AC (invariantes a DC) — usáveis em ML.
FEATS_AC = ["std", "pico_abs", "pico_pico", "crista", "assimetria", "curtose"]
# Carregam DC (offset/sessão). Só DIAGNÓSTICO. potencia = rms^2.
FEATS_DC = ["media", "rms", "potencia"]
FEATS = FEATS_DC + FEATS_AC

# Repetição usada para o par representativo e para toda a análise.
REP_PAR = 1


# ----------------------------------------------------------------------------
# Extração
# ----------------------------------------------------------------------------
def extract_features(W):
    """W: (janelas, eixos, amostras) -> dict de arrays (janelas, eixos)."""
    W = np.asarray(W, dtype=np.float64)
    mu = W.mean(axis=2)
    Wc = W - mu[..., None]
    std = Wc.std(axis=2)
    peak = np.abs(Wc).max(axis=2)
    pot = (W ** 2).mean(axis=2)
    return {
        "media": mu, "rms": np.sqrt(pot), "potencia": pot, "std": std,
        "pico_abs": peak, "pico_pico": np.ptp(W, axis=2),
        "crista": peak / std,
        "assimetria": skew(Wc, axis=2),
        "curtose": kurtosis(Wc, axis=2, fisher=False),
    }


def build_feature_table(sensor):
    W = np.load(OUT_DIR / f"janelas_{sensor}.npy", mmap_mode="r")
    idx = pd.read_csv(OUT_DIR / f"janelas_{sensor}_indice.csv")

    # Metadados vivem em registros_metadados.csv; recupera por record_id.
    meta = pd.read_csv(OUT_DIR / "registros_metadados.csv")
    meta = meta[meta.sensor == sensor][
        ["record_id", "fault", "severity", "speed", "channel", "repetition"]
    ].drop_duplicates("record_id")
    idx = idx.merge(meta, on="record_id", how="left")

    F = extract_features(W)
    parts = []
    for a, ax in enumerate(AXES[sensor]):
        d = idx.copy()
        d["eixo"] = ax
        for k in FEATS:
            d[k] = F[k][:, a]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["classe"] = np.where(df["fault"] == 0, "saudável", "defeituoso")
    return df


def escolher_par(df, sensor, repetition=REP_PAR):
    """Par representativo: std mediano mais próximo da mediana da classe,
    restrito a uma repetição (mesma sessão)."""
    ax0 = AXES[sensor][0]
    d = df[(df.eixo == ax0) & (df.speed == PAR_SP)
           & (df.channel == CANAL_RELATORIO) & (df.repetition == repetition)]
    med = d.groupby("record_id")["std"].median()
    classe = d.groupby("record_id").classe.first()
    par = []
    for c in ("saudável", "defeituoso"):
        m = med[classe == c]
        if m.empty:
            return None
        par.append((m - m.median()).abs().idxmin())
    return tuple(par)


# ----------------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------------
def effect_sizes(df, keys):
    """AUC (Mann-Whitney) e d de Cohen, defeituoso vs saudável.
    Janelas do mesmo registro não são independentes: indicativo, não teste."""
    rows = []
    for k, d in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        a, b = d[d.classe == "defeituoso"], d[d.classe == "saudável"]
        if a.empty or b.empty:
            continue
        for f in FEATS:
            u = mannwhitneyu(a[f], b[f]).statistic
            auc = u / (len(a) * len(b))
            sp = np.sqrt((a[f].var() + b[f].var()) / 2)
            rows.append(dict(zip(keys, k), feature=f, auc=auc,
                             auc_sep=max(auc, 1 - auc),
                             cohen_d=(a[f].mean() - b[f].mean()) / sp))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    par = None
    dfs = []
    for sensor in AXES:
        if not (OUT_DIR / f"janelas_{sensor}.npy").exists():
            continue
        df = build_feature_table(sensor)
        df["sensor"] = sensor
        dfs.append(df)

        if PLOT:
            from graph_plot2 import (plot_time_comparison, plot_boxplots,
                                      plot_by_speed, plot_corr)
            if par is None:
                par = escolher_par(df, sensor)
                print(f"Par temporal (SP{PAR_SP}, C{CANAL_RELATORIO}, "
                      f"R{REP_PAR}): {par}")
            if par:
                plot_time_comparison(sensor, par)
            plot_boxplots(df, sensor, detalhado=False)
            if sensor == "ACC":
                plot_by_speed(df[df.channel == CANAL_RELATORIO], sensor)
                plot_corr(df, sensor)

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_DIR / "features_tempo_longo.csv", index=False)

    # Toda a análise numérica é restrita a R1 (mesma sessão).
    full_r1 = full[full.repetition == REP_PAR]

    # Verificação da identidade RMS^2 = media^2 + std^2 (item 3.2.4)
    erro = (full_r1["rms"] ** 2 - (full_r1["media"] ** 2 + full_r1["std"] ** 2)).abs().max()
    print(f"\nmax |RMS² - (média² + std²)| = {erro:.2e}   (deve ser ~0)")
    print(f"max |potência - RMS²|        = "
          f"{(full_r1['potencia'] - full_r1['rms'] ** 2).abs().max():.2e}")

    # Separabilidade por velocidade
    eff = effect_sizes(full_r1, ["sensor", "eixo", "speed"])
    eff.round(3).to_csv(OUT_DIR / "tamanho_efeito_r1_por_velocidade.csv",
                        index=False)
    print(f"\n=== Separabilidade por velocidade, R{REP_PAR} (AUC simétrica) ===")
    print(eff.pivot_table(index=["sensor", "eixo", "speed"],
                          columns="feature", values="auc_sep")
          .round(2).to_string())

    # Resumo (média/desvio) por classe e velocidade
    resumo = full_r1.groupby(["sensor", "eixo", "speed", "classe"])[FEATS].agg(
        ["mean", "std"]).round(4)
    resumo.to_csv(OUT_DIR / "tabela_resumo_velocidade_r1.csv")
    print(f"\n=== Resumo R{REP_PAR} por velocidade (média das janelas) ===")
    print(resumo.to_string())