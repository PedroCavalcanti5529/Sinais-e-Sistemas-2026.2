"""
Espectrograma (STFT) dos sinais de vibração — complementa a FFT global da 3.3.

O que a FFT de uma janela de 4 s não mostra: COMO o espectro varia no tempo. O
espectrograma divide o registro em segmentos curtos (janela de Hann), calcula a FFT de
cada um e empilha as PSDs: eixo x = tempo, eixo y = frequência, cor = PSD em dB.
Serve para (i) ver partida/transitórios e deriva de rotação, (ii) conferir se a hipótese
de estacionariedade da FFT de 4 s é razoável, (iii) comparar saudável × defeituoso em
tempo-frequência, (iv) mostrar o compromisso resolução em tempo × resolução em frequência.

Leitura: os .mat originais (DRONE_DATA_DIR) pelos MESMOS leitores/pré-processamento da
3.1 (preprocess3_1.load_mat / get_sensor / preprocess), a partir de registros_metadados.csv.
Usa o registro contínuo (as janelas de 4 s da 3.1 têm 50% de overlap e não são contíguas).

Parâmetros da STFT (padrão): janela de Hann de 0,25 s (Δf = fs/nperseg = 4 Hz a 10 kHz),
overlap 75% (passo de 62,5 ms), detrend constante por segmento (remove o offset/DC),
PSD unilateral em g²/Hz exibida em dB.

Uso:
  python espectrograma.py                       # grade, resolução e estacionariedade
  python espectrograma.py F0_SV0_SP1_C1_R1 ...  # espectrograma + potência por banda
                                                  de registros específicos

Saídas (OUT_DIR/figuras_espectrograma/ e OUT_DIR):
  grade_espectrogramas_<sensor>.png          linhas = condição, colunas = velocidade
  resolucao_tempo_frequencia_<sensor>.png    mesma STFT com 3 durações de janela
  espectrograma_<record_id>.png              (modo por registro)
  estacionariedade_espectrograma.csv         CV da potência por banda e deriva do centroide
  estacionariedade_cv.png                    boxplots do CV por classe
"""
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from config import DATA_DIR, OUT_DIR, REP_PAR, PAR_SP, PAR_CANAL, BANDAS, UNIT_SHORT
from preprocess3_1 import load_mat, get_sensor, preprocess

SENSOR = "ACC"
JANELA_S = 0.25            # janela de Hann principal [s]
OVERLAP = 0.75
JANELAS_RESOLUCAO = (0.1, 0.25, 1.0)
FMAX_PLOT = 1500.0         # limite superior do eixo de frequência nas figuras [Hz]
DINAMICA_DB = 60.0         # faixa dinâmica da escala de cor (vmax - DINAMICA_DB)
JANELA_ESTAC_S = 0.5       # janela usada no estudo de estacionariedade
CV_ESTAC_ALERTA = 0.25     # CV da banda B2 acima disso = potência pouco estacionária

FIG_DIR = OUT_DIR / "figuras_espectrograma"
CORES = {"saudável": "tab:blue", "defeituoso": "tab:red"}


# ----------------------------------------------------------------------------
# Leitura e STFT
# ----------------------------------------------------------------------------
def carregar_metadados():
    meta = pd.read_csv(OUT_DIR / "registros_metadados.csv")
    meta = meta[meta.sensor == SENSOR].drop_duplicates("record_id")
    meta["classe"] = np.where(meta.fault == 0, "saudável", "defeituoso")
    return meta.reset_index(drop=True)


@lru_cache(maxsize=32)
def carregar_sinal(arquivo):
    """(x, fs, offset_s): eixo 0 do sensor após o pré-processamento da 3.1."""
    path = Path(DATA_DIR) / arquivo
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Defina DRONE_DATA_DIR.")
    X, _, fs = get_sensor(load_mat(path), SENSOR)
    Y, off_s, _, _ = preprocess(X, fs)
    return Y[0], fs, off_s


def calc_espectrograma(x, fs, janela_s=JANELA_S, overlap=OVERLAP):
    """STFT com janela de Hann. Retorna f [Hz], t [s] (centros), S [unidade²/Hz]."""
    nper = int(round(janela_s * fs))
    nper = min(nper, len(x))
    nov = int(nper * overlap)
    f, t, S = spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nov,
                          detrend="constant", scaling="density", mode="psd")
    return f, t, S


def para_db(S):
    return 10.0 * np.log10(S + 1e-20)


def potencia_bandas(f, S):
    """Série temporal da potência por banda (unidade²), sem o bin de DC."""
    df = f[1] - f[0]
    out = {}
    for nome, lo, hi in BANDAS:
        hi_eff = min(hi, f[-1])
        m = (f > 0) & (f >= lo) & ((f <= hi_eff) if hi >= f[-1] else (f < hi_eff))
        out[nome] = S[m].sum(axis=0) * df if m.any() else np.zeros(S.shape[1])
    return out


# ----------------------------------------------------------------------------
# Seleção de registros para a comparação
# ----------------------------------------------------------------------------
def selecionar_condicoes(meta):
    """Por velocidade: saudável + (para cada tipo de falha) a maior severidade.
    Mesmo canal e repetição da análise principal."""
    m = meta[(meta.repetition == REP_PAR) & (meta.channel == PAR_CANAL)]
    condicoes = {}
    for sp, g in m.groupby("speed"):
        linhas = []
        h = g[g.fault == 0]
        if len(h):
            linhas.append(("saudável", h.iloc[0]))
        for fl, gf in g[g.fault != 0].groupby("fault"):
            r = gf.sort_values("severity").iloc[-1]
            linhas.append((f"F{int(fl)} SV{int(r.severity)}", r))
        condicoes[sp] = linhas
    return condicoes


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def plot_grade(meta):
    """Linhas = condição (saudável, F1, F2, ...), colunas = velocidade; escala de cor
    comum a todos os painéis para a comparação ser legítima."""
    cond = selecionar_condicoes(meta)
    speeds = sorted(cond)
    rotulos = []
    for sp in speeds:
        for rot, _ in cond[sp]:
            if rot.split()[0] not in [r.split()[0] for r in rotulos]:
                rotulos.append(rot)
    chave = lambda r: r.split()[0]

    dados = {}
    for sp in speeds:
        for rot, row in cond[sp]:
            x, fs, off = carregar_sinal(row.file)
            f, t, S = calc_espectrograma(x, fs)
            k = f <= FMAX_PLOT
            dados[(chave(rot), sp)] = (f[k], t + off, para_db(S[k]), rot, row.record_id)
    if not dados:
        print("[AVISO] nenhuma condição encontrada para a grade.")
        return
    vmax = np.percentile(np.concatenate([v[2].ravel() for v in dados.values()]), 99.5)
    vmin = vmax - DINAMICA_DB

    nl, nc = len(rotulos), len(speeds)
    fig, axs = plt.subplots(nl, nc, figsize=(5.2 * nc, 2.6 * nl), sharey=True,
                            squeeze=False, constrained_layout=True)
    im = None
    for i, rot in enumerate(rotulos):
        for j, sp in enumerate(speeds):
            ax = axs[i, j]
            v = dados.get((chave(rot), sp))
            if v is None:
                ax.axis("off")
                continue
            f, t, D, rot_real, rid = v
            im = ax.pcolormesh(t, f, D, shading="auto", vmin=vmin, vmax=vmax,
                               cmap="magma", rasterized=True)
            ax.set_title(f"{rot_real} — SP{sp} ({rid})", fontsize=8)
            if j == 0:
                ax.set_ylabel("frequência [Hz]")
            if i == nl - 1:
                ax.set_xlabel("tempo [s]")
    fs0 = carregar_sinal(cond[speeds[0]][0][1].file)[1]
    nper = int(round(JANELA_S * fs0))
    fig.colorbar(im, ax=axs.ravel().tolist(),
                 label=f"PSD [dB re 1 {UNIT_SHORT[SENSOR]}²/Hz]")
    fig.suptitle(f"Espectrogramas {SENSOR} (Hann {JANELA_S * 1000:.0f} ms, overlap "
                 f"{OVERLAP:.0%}, Δf = {fs0 / nper:.1f} Hz; canal C{PAR_CANAL}, R{REP_PAR})")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"grade_espectrogramas_{SENSOR}.png", dpi=130)
    plt.close(fig)


def plot_resolucao(meta):
    """Compromisso tempo × frequência: janela curta → boa resolução temporal e Δf
    grosseiro; janela longa → Δf fino e transitórios borrados."""
    cond = selecionar_condicoes(meta).get(PAR_SP, [])
    if len(cond) < 2:
        print("[AVISO] sem par saudável/defeituoso em PAR_SP; pulando resolução.")
        return
    escolhidos = [cond[0], cond[1]]                    # saudável e 1º defeituoso
    dados = {}
    for lin, (rot, row) in enumerate(escolhidos):
        x, fs, off = carregar_sinal(row.file)
        for c, js in enumerate(JANELAS_RESOLUCAO):
            f, t, S = calc_espectrograma(x, fs, janela_s=js)
            k = f <= FMAX_PLOT
            dados[(lin, c)] = (f[k], t + off, para_db(S[k]), fs, js)
    vmax = np.percentile(np.concatenate([v[2].ravel() for v in dados.values()]), 99.5)
    fig, axs = plt.subplots(2, len(JANELAS_RESOLUCAO), figsize=(5.2 * len(JANELAS_RESOLUCAO), 6),
                            sharey=True, squeeze=False, constrained_layout=True)
    im = None
    for (lin, c), (f, t, D, fs, js) in dados.items():
        ax = axs[lin, c]
        im = ax.pcolormesh(t, f, D, shading="auto", vmin=vmax - DINAMICA_DB, vmax=vmax,
                           cmap="magma", rasterized=True)
        nper = int(round(js * fs))
        ax.set_title(f"{escolhidos[lin][0]} — janela {js * 1000:.0f} ms\n"
                     f"Δf = {fs / nper:.1f} Hz, passo = {js * (1 - OVERLAP) * 1000:.0f} ms",
                     fontsize=9)
        if c == 0:
            ax.set_ylabel("frequência [Hz]")
        if lin == 1:
            ax.set_xlabel("tempo [s]")
    fig.colorbar(im, ax=axs.ravel().tolist(), label=f"PSD [dB re 1 {UNIT_SHORT[SENSOR]}²/Hz]")
    fig.suptitle(f"Resolução tempo × frequência — {SENSOR}, SP{PAR_SP}, C{PAR_CANAL}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"resolucao_tempo_frequencia_{SENSOR}.png", dpi=130)
    plt.close(fig)


def plot_registro(meta, record_id):
    """Espectrograma de um registro + potência por banda ao longo do tempo."""
    r = meta[meta.record_id == record_id]
    if r.empty:
        print(f"[AVISO] registro {record_id} não está em registros_metadados.csv.")
        return
    row = r.iloc[0]
    x, fs, off = carregar_sinal(row.file)
    f, t, S = calc_espectrograma(x, fs)
    t = t + off
    k = f <= FMAX_PLOT
    D = para_db(S[k])
    vmax = np.percentile(D, 99.5)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]),
                                 constrained_layout=True)
    im = a1.pcolormesh(t, f[k], D, shading="auto", vmin=vmax - DINAMICA_DB, vmax=vmax,
                       cmap="magma", rasterized=True)
    a1.set_ylabel("frequência [Hz]")
    a1.set_title(f"{record_id} — {row.classe} (Hann {JANELA_S * 1000:.0f} ms, "
                 f"Δf = {fs / int(round(JANELA_S * fs)):.1f} Hz)")
    fig.colorbar(im, ax=a1, label=f"PSD [dB re 1 {UNIT_SHORT[SENSOR]}²/Hz]")
    for nome, p in potencia_bandas(f, S).items():
        a2.plot(t, 10 * np.log10(p + 1e-20), lw=0.9, label=nome)
    a2.set_xlabel("tempo [s]")
    a2.set_ylabel(f"potência da banda [dB re 1 {UNIT_SHORT[SENSOR]}²]")
    a2.grid(True, alpha=0.3)
    a2.legend(ncol=4, fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"espectrograma_{record_id}.png", dpi=130)
    plt.close(fig)


def estacionariedade(meta):
    """CV (std/média) da potência por banda e do centroide ao longo do tempo, por
    registro R1. CV baixo = espectro estável, coerente com a FFT de 4 s; CV alto =
    transitórios/deriva, e as features por janela de 4 s misturam regimes."""
    rows = []
    for _, row in meta[meta.repetition == REP_PAR].iterrows():
        x, fs, off = carregar_sinal(row.file)
        f, t, S = calc_espectrograma(x, fs, janela_s=JANELA_ESTAC_S, overlap=0.5)
        pb = potencia_bandas(f, S)
        S_sem = S[1:]
        cent = (f[1:, None] * S_sem).sum(axis=0) / (S_sem.sum(axis=0) + 1e-30)
        d = dict(record_id=row.record_id, classe=row.classe, fault=row.fault,
                 severity=row.severity, speed=row.speed, channel=row.channel,
                 duracao_s=round(len(x) / fs, 1))
        for nome, p in pb.items():
            d[f"cv_{nome}"] = float(p.std() / (p.mean() + 1e-30))
        d["cv_centroide"] = float(cent.std() / (cent.mean() + 1e-30))
        rows.append(d)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT_DIR / "estacionariedade_espectrograma.csv", index=False)

    cols = [c for c in df.columns if c.startswith("cv_")]
    fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.8))
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, cols):
        dados = [df[df.classe == k][c].values for k in CORES]
        try:
            bp = ax.boxplot(dados, tick_labels=list(CORES), patch_artist=True)
        except TypeError:
            bp = ax.boxplot(dados, labels=list(CORES), patch_artist=True)
        for p, col in zip(bp["boxes"], CORES.values()):
            p.set_facecolor(col); p.set_alpha(0.5)
        ax.set_title(c.replace("cv_", "CV "), fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    axs[0].set_ylabel("std / média ao longo do tempo")
    fig.suptitle(f"Estacionariedade (janela {JANELA_ESTAC_S * 1000:.0f} ms), R{REP_PAR}")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "estacionariedade_cv.png", dpi=130)
    plt.close(fig)

    if "cv_B2" in df:
        n_alerta = int((df.cv_B2 > CV_ESTAC_ALERTA).sum())
        print(f"\nEstacionariedade (R{REP_PAR}): {n_alerta}/{len(df)} registros com "
              f"CV(B2) > {CV_ESTAC_ALERTA}.")
        print(df.groupby("classe")[cols].median().round(3).to_string())
    return df


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    meta = carregar_metadados()
    alvos = sys.argv[1:]
    if alvos:
        for rid in alvos:
            plot_registro(meta, rid)
            print(f"salvo: espectrograma_{rid}.png")
    else:
        plot_grade(meta)
        plot_resolucao(meta)
        estacionariedade(meta)
        print(f"Figuras em {FIG_DIR}")
