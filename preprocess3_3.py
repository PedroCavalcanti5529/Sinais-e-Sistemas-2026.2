"""
Parte 3.3 - Análise espectral (FFT) dos sinais de vibração  (v3)

Entrada : janelas_{ACC,IMU}.npy + janelas_{...}_indice.csv + registros_metadados.csv
          (janelas de 4 s da 3.1, 50% de overlap)
Saídas  : features_espectrais_longo.csv      (uma linha por janela × eixo)
          tabela_espectral_resumo.csv        (média/desvio por classe e velocidade;
                                              colunas achatadas <feature>_mean/_std)
          tamanho_efeito_espectral.csv       (AUC por feature/sensor/eixo/velocidade)
          picos_por_condicao.csv             (NOVO: top-5 picos por velocidade × falha)
          auc_por_frequencia.csv             (NOVO: AUC por sub-banda de 5 Hz)
          figuras_espectrais/espectros_*.png        (item 2; versões log e linear c/ picos)
          figuras_espectrais/espectros_por_falha_*.png  (NOVO)
          figuras_espectrais/boxplots_ACC.png       (item 4)
          figuras_espectrais/dose_resposta_*.png    (NOVO: feature × severidade)
          figuras_espectrais/auc_por_frequencia_*.png (NOVO)
          figuras_espectrais/efeito_duracao_*.png / _zoom.png   (item 6)
          figuras_espectrais/efeito_aliasing_*.png  (item 7: vários fatores + alias previsto)
          figuras_espectrais/periodograma_vs_welch_*.png (NOVO)

Normalização adotada (janela retangular: sem janelamento, Parseval exato)
  - Amplitude (pico da senoide equivalente): A[k] = 2 * |X[k]| / N  (k > 0)
    A[0] = |X[0]| / N  (DC, sem fator 2)
  - PSD unilateral: P[k] = 2 * |X[k]|^2 / (fs * N)  (k > 0), P[0] = |X[0]|^2 / (fs*N)
    Integrar P[k] * Δf em uma banda dá a potência média (unidade^2) na banda.
  - Limitação: a janela retangular vaza energia dos picos para bins vizinhos (e eleva
    o piso). Aqui é uma escolha didática para a FFT do item 3.3.1; o Welch com Hann é
    mostrado em periodograma_vs_welch_*.png.

Bandas — o bin de DC (f = 0) é EXCLUÍDO de todas as bandas (carrega o offset da sessão).
  - B1 = (0, 50] Hz   : movimentos estruturais/rolagem; abaixo da dinâmica de hélice.
  - B2 = 50–300 Hz    : dinâmica de hélice (rotação x nº de pás).
  - B3 = 300–1000 Hz  : harmônicos e ressonâncias de alta ordem.
  - B4 = 1000–5000 Hz : (NOVA) cauda de alta frequência até Nyquist; antes ficava fora das
                        bandas, embora centroide/entropia usassem o espectro todo.
  Potência RELATIVA (potrel_Bk = potência na banda / potência total sem DC) mede o formato
  do espectro e não a energia (que std/RMS já medem). Σ potrel ≈ 1 → uma delas é redundante.

Features novas: planicidade (Wiener: média geométrica/aritmética da PSD) e rolloff85
(frequência abaixo da qual há 85% da potência).

Notas de confundimento: R2/R3 são de outra sessão; a análise principal usa R1.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly, welch
from scipy.stats import mannwhitneyu

from config import (OUT_DIR, AXES, UNIT_SHORT, ACTIVE_SENSORS, REP_PAR, PAR_SP,
                    PAR_CANAL, BANDAS, FEATS_ESPECTRAIS)
from utils import auc_rank

FIG_DIR = OUT_DIR / "figuras_espectrais"

CORES = {"saudável": "tab:blue", "defeituoso": "tab:red"}
CHUNK = 250                         # janelas por bloco (limita memória)
FMAX_PLOT = 1000.0                  # zoom linear dos espectros [Hz]
FMAX_PICOS = 2000.0                 # busca de picos até [Hz]

# Item 6 (efeito da duração): recorta 1 s da janela de 4 s
DURACOES_COMPARADAS = (1.0, 4.0)
# Item 7 (aliasing): fatores inteiros de decimação. Fatores maiores (10, 20) empurram
# os picos principais para além do novo Nyquist e tornam o alias visível.
FATORES_DECIMACAO = {"ACC": (5, 10, 20), "IMU": (2,)}


# ----------------------------------------------------------------------------
# Carregamento (mesmo contrato da 3.2)
# ----------------------------------------------------------------------------
def carregar_sensor(sensor, repetition=None, so_saudavel=False):
    """Retorna (W, meta_por_janela). W: (n_janelas, n_eixos, n_amostras)."""
    W = np.load(OUT_DIR / f"janelas_{sensor}.npy", mmap_mode="r")
    idx = pd.read_csv(OUT_DIR / f"janelas_{sensor}_indice.csv")
    meta = pd.read_csv(OUT_DIR / "registros_metadados.csv")
    meta = meta[meta.sensor == sensor][
        ["record_id", "fault", "severity", "speed", "channel", "repetition"]
    ].drop_duplicates("record_id")
    idx = idx.merge(meta, on="record_id", how="left")
    idx["classe"] = np.where(idx["fault"] == 0, "saudável", "defeituoso")
    mask = np.ones(len(idx), dtype=bool)
    if repetition is not None:
        mask &= (idx["repetition"] == repetition).to_numpy()
    if so_saudavel:
        mask &= (idx["fault"] == 0).to_numpy()
    if not mask.all():
        W = W[mask]
        idx = idx[mask].reset_index(drop=True)
    return W, idx


def primeiro_registro(idx, classe, speed=PAR_SP, canal=PAR_CANAL):
    """(record_id, posição da 1ª janela) do primeiro registro da classe,
    mesma velocidade e canal para as duas classes (comparável)."""
    m = ((idx["classe"] == classe) & (idx["speed"] == speed)
         & (idx["channel"] == canal)).to_numpy()
    if not m.any():
        return None, None
    pos = int(np.argmax(m))
    return idx["record_id"].iloc[pos], pos


# ----------------------------------------------------------------------------
# FFT e features espectrais
# ----------------------------------------------------------------------------
def fft_amplitude(x, fs):
    """Amplitude unilateral (pico da senoide equivalente) e eixo de f.
    x: (n_amostras,) ou (n_janelas, n_amostras)."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    A = 2.0 * np.abs(X) / N
    A[..., 0] /= 2.0                      # DC sem fator 2
    return A, f


def psd_unilateral(x, fs):
    """PSD unilateral (unidade^2 / Hz). Integra com Δf -> potência média."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    P = 2.0 * np.abs(X) ** 2 / (fs * N)
    P[..., 0] /= 2.0
    return P


def _mascara_banda(f, lo, hi, fs):
    """Banda [lo, hi); a última (hi >= Nyquist) é fechada em Nyquist."""
    hi_eff = min(hi, fs / 2.0)
    sup = (f <= hi_eff) if hi >= fs / 2.0 else (f < hi_eff)
    return (f > 0) & (f >= lo) & sup


def features_espectrais_janela(W_eixo, fs):
    """W_eixo: (n_janelas, n_amostras) -> dict de arrays (n_janelas,)."""
    A, f = fft_amplitude(W_eixo, fs)          # (n_janelas, n_freq)
    P = psd_unilateral(W_eixo, fs)
    df = f[1] - f[0]

    # Todas as medidas ignoram o DC para não pegar o offset
    A_sem_dc, P_sem_dc, f_sem_dc = A[:, 1:], P[:, 1:], f[1:]
    i_dom = A_sem_dc.argmax(axis=1)
    freq_dom = f_sem_dc[i_dom]
    mag_dom = A_sem_dc[np.arange(len(i_dom)), i_dom]

    # Centroide espectral (ponderado por amplitude)
    soma_A = A_sem_dc.sum(axis=1) + 1e-30
    centroide = (f_sem_dc * A_sem_dc).sum(axis=1) / soma_A

    # Entropia espectral (normalizada em [0, 1]; p log p com p = A / sum A)
    p = A_sem_dc / soma_A[:, None]
    entropia = -(p * np.log2(p + 1e-30)).sum(axis=1) / np.log2(len(f_sem_dc))

    # Largura de banda efetiva: 2 * sqrt(variancia espectral em torno do centroide)
    var = ((f_sem_dc[None, :] - centroide[:, None]) ** 2 * A_sem_dc).sum(axis=1) / soma_A
    largura = 2.0 * np.sqrt(var)

    # Planicidade espectral (Wiener) sobre a PSD: 1 = ruído branco, 0 = tonal
    planic = (np.exp(np.log(P_sem_dc + 1e-30).mean(axis=1))
              / (P_sem_dc.mean(axis=1) + 1e-30))

    # Rolloff: menor f com 85% da potência acumulada
    cum = np.cumsum(P_sem_dc, axis=1)
    rolloff = f_sem_dc[(cum >= 0.85 * cum[:, -1:]).argmax(axis=1)]

    out = {"freq_dominante": freq_dom, "mag_dominante": mag_dom,
           "centroide": centroide, "entropia": entropia,
           "largura_banda": largura, "planicidade": planic, "rolloff85": rolloff}

    # Potência em bandas (integração da PSD, unidade^2), sem o bin de DC
    total = P_sem_dc.sum(axis=1) * df + 1e-30
    for nome, lo, hi in BANDAS:
        pot = P[:, _mascara_banda(f, lo, hi, fs)].sum(axis=1) * df
        out[f"potencia_{nome}"] = pot
        out[f"potrel_{nome}"] = pot / total
    return out


def tabela_features_espectrais(W, idx, sensor):
    """W: (n_janelas, n_eixos, n_amostras) -> DataFrame longo (processa em blocos)."""
    fs = float(idx["fs_hz"].iloc[0]) if len(idx) else 1.0
    partes = []
    for a, ax in enumerate(AXES[sensor]):
        blocos = [features_espectrais_janela(np.asarray(W[i:i + CHUNK, a, :]), fs)
                  for i in range(0, len(W), CHUNK)]
        F = {k: np.concatenate([b[k] for b in blocos]) for k in blocos[0]}
        d = idx.copy()
        d["sensor"] = sensor
        d["eixo"] = ax
        for k, v in F.items():
            d[k] = v
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------------
def effect_sizes_espectral(df, keys):
    rows = []
    for k, d in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        a, b = d[d.classe == "defeituoso"], d[d.classe == "saudável"]
        if a.empty or b.empty:
            continue
        for f in FEATS_ESPECTRAIS:
            u = mannwhitneyu(a[f], b[f]).statistic
            auc = u / (len(a) * len(b))
            sp = np.sqrt((a[f].var() + b[f].var()) / 2) + 1e-30
            rows.append(dict(zip(keys, k), feature=f, auc=auc,
                             auc_sep=max(auc, 1 - auc),
                             cohen_d=(a[f].mean() - b[f].mean()) / sp))
    return pd.DataFrame(rows)


def _picos(A, f, k=3, rel=0.05, fmax=FMAX_PICOS):
    """Até k picos (f, amplitude, proeminência) do espectro A, f>1 Hz, por proeminência."""
    m = (f > 1.0) & (f <= fmax)
    fm, Am = f[m], A[m]
    if Am.size < 3 or Am.max() <= 0:
        return []
    pk, props = find_peaks(Am, prominence=rel * Am.max())
    ordem = np.argsort(props["prominences"])[::-1][:k]
    return [(float(fm[pk[i]]), float(Am[pk[i]]), float(props["prominences"][i]))
            for i in ordem]


def tabela_picos(W, idx, sensor, canal=PAR_CANAL, k=5):
    """Top-k picos do espectro médio de amplitude, por velocidade × falha (canal fixo).
    Falha 0 = saudável. Média sobre todas as severidades da mesma falha."""
    fs = float(idx["fs_hz"].iloc[0])
    rows = []
    for sp in sorted(idx["speed"].unique()):
        for fl in sorted(idx["fault"].unique()):
            m = ((idx["speed"] == sp) & (idx["fault"] == fl)
                 & (idx["channel"] == canal)).to_numpy()
            if not m.any():
                continue
            A, f = fft_amplitude(np.asarray(W[m, 0, :]), fs)
            for r, (fp, ap, pr) in enumerate(_picos(A.mean(axis=0), f, k=k), 1):
                rows.append(dict(sensor=sensor, speed=sp, fault=fl,
                                 classe="saudável" if fl == 0 else "defeituoso",
                                 rank=r, freq_hz=round(fp, 2),
                                 amplitude=ap, proeminencia=pr))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figuras (itens 2, 4, 6, 7 + extras)
# ----------------------------------------------------------------------------
def plot_espectros_comparativos(W, idx, sensor, canal=None):
    """Item 2: amplitude média por classe e velocidade. Gera versão log (visão geral)
    e versão linear 0–FMAX_PLOT Hz com os 3 picos mais proeminentes anotados."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    speeds = sorted(idx["speed"].unique())
    for a, ax in enumerate(AXES[sensor]):
        medias = {}
        for sp in speeds:
            for classe in CORES:
                m = (idx["speed"] == sp) & (idx["classe"] == classe)
                if canal is not None:
                    m &= (idx["channel"] == canal)
                if m.any():
                    A, f = fft_amplitude(np.asarray(W[m.to_numpy(), a, :]), fs)
                    medias[(sp, classe)] = (f, A.mean(axis=0))

        for modo in ("log", "linear"):
            fig, eixos = plt.subplots(1, len(speeds), figsize=(5 * len(speeds), 4),
                                      sharey=True, squeeze=False)
            for j, sp in enumerate(speeds):
                axp = eixos[0, j]
                for i, (classe, cor) in enumerate(CORES.items()):
                    if (sp, classe) not in medias:
                        continue
                    f, Am = medias[(sp, classe)]
                    axp.semilogy(f[1:], Am[1:], color=cor, label=classe, lw=1)
                    if modo == "linear":
                        for fp, ap, _ in _picos(Am, f, k=3, fmax=FMAX_PLOT):
                            axp.annotate(f"{fp:.0f} Hz", (fp, ap), fontsize=7,
                                         xytext=(0, 5 + 9 * i), color=cor,
                                         textcoords="offset points", ha="center")
                axp.set_title(f"{sensor}/{ax} — SP{sp}")
                axp.set_xlabel("frequência [Hz]")
                if modo == "log":
                    axp.set_xscale("log")
                else:
                    axp.set_xlim(0, FMAX_PLOT)
                axp.grid(True, which="both", alpha=0.3)
                if j == 0:
                    axp.set_ylabel(f"amplitude média [{un}]")
            eixos[0, 0].legend()
            sufixo = f"_C{canal}" if canal is not None else ""
            fig.tight_layout()
            nome = f"espectros_{sensor}_{ax}{sufixo}" + ("_linear" if modo == "linear" else "")
            fig.savefig(FIG_DIR / f"{nome}.png", dpi=120)
            plt.close(fig)


def plot_espectros_por_falha(W, idx, sensor, canal=PAR_CANAL):
    """Espectro médio por TIPO de falha (F1, F2, ...) + saudável, por velocidade.
    O 'defeituoso' binário mistura configurações e borra os picos; aqui separa."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    speeds = sorted(idx["speed"].unique())
    faults = sorted(idx["fault"].unique())
    cmap = plt.get_cmap("tab10")
    fig, eixos = plt.subplots(1, len(speeds), figsize=(5 * len(speeds), 4),
                              sharey=True, squeeze=False)
    for j, sp in enumerate(speeds):
        axp = eixos[0, j]
        for fl in faults:
            m = ((idx["speed"] == sp) & (idx["fault"] == fl)
                 & (idx["channel"] == canal)).to_numpy()
            if not m.any():
                continue
            A, f = fft_amplitude(np.asarray(W[m, 0, :]), fs)
            axp.semilogy(f[1:], A[:, 1:].mean(axis=0), lw=0.9, color=cmap(fl),
                         label="saudável" if fl == 0 else f"F{fl} (todas as SV)")
        axp.set_xlim(0, FMAX_PLOT)
        axp.set_title(f"{sensor} — SP{sp}, C{canal}")
        axp.set_xlabel("frequência [Hz]")
        axp.grid(True, which="both", alpha=0.3)
        if j == 0:
            axp.set_ylabel(f"amplitude média [{un}]")
    eixos[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"espectros_por_falha_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_efeito_duracao(W, idx, sensor):
    """Item 6: mesma janela (4 s) analisada inteira e recortada em 1 s.
    Linhas = classe, colunas = duração. Δf = fs/N."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    dur_total = W.shape[-1] / fs
    duracoes = [d for d in DURACOES_COMPARADAS if d <= dur_total + 1e-9]

    fig, eixos = plt.subplots(2, len(duracoes), figsize=(6 * len(duracoes), 7),
                              sharey=True, squeeze=False)
    for i, (classe, cor) in enumerate(CORES.items()):
        rid, pos = primeiro_registro(idx, classe)
        if rid is None:
            plt.close(fig)
            return
        x_full = np.asarray(W[pos, 0, :], dtype=np.float64)
        for j, dur in enumerate(duracoes):
            x = x_full[: int(round(fs * dur))]
            A, f = fft_amplitude(x, fs)
            axp = eixos[i, j]
            axp.semilogy(f[1:], A[1:], color=cor, lw=0.8)
            axp.set_title(f"{rid} ({classe}) — {dur:.0f} s, N={len(x)}, "
                          f"Δf={f[1]:.3f} Hz", fontsize=9)
            axp.grid(True, which="both", alpha=0.3)
            if i == 1:
                axp.set_xlabel("frequência [Hz]")
            if j == 0:
                axp.set_ylabel(f"amplitude [{un}]")
    fig.suptitle(f"Item 6: efeito da duração — {sensor}/{AXES[sensor][0]}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"efeito_duracao_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_efeito_duracao_zoom(W, idx, sensor, meia_largura_hz=15.0):
    """Item 6 (complemento): 4 s × 1 s sobrepostos, linear, em torno do pico dominante.
    Os marcadores mostram os bins reais: espaçamento Δf = fs/N visível."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    rid, pos = primeiro_registro(idx, "saudável")
    if rid is None:
        return
    x_full = np.asarray(W[pos, 0, :], dtype=np.float64)
    A4, f4 = fft_amplitude(x_full, fs)
    f0 = f4[1:][A4[1:].argmax()]
    fig, axp = plt.subplots(figsize=(8, 4.2))
    for dur, mk, cor in zip(sorted(DURACOES_COMPARADAS)[::-1], ("o-", "s-"),
                            ("tab:blue", "tab:orange")):
        if dur > W.shape[-1] / fs + 1e-9:
            continue
        x = x_full[: int(round(fs * dur))]
        A, f = fft_amplitude(x, fs)
        m = (f >= f0 - meia_largura_hz) & (f <= f0 + meia_largura_hz)
        axp.plot(f[m], A[m], mk, ms=4, lw=0.8, color=cor,
                 label=f"{dur:.0f} s (N={len(x)}, Δf={f[1]:.2f} Hz)")
    axp.set_xlabel("frequência [Hz]")
    axp.set_ylabel(f"amplitude [{un}]")
    axp.set_title(f"Zoom no pico dominante (~{f0:.1f} Hz) — {rid}")
    axp.grid(True, alpha=0.3)
    axp.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"efeito_duracao_zoom_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_efeito_aliasing(W, idx, sensor):
    """Item 7: para cada fator, decimação SEM filtro (aliasing real) e COM filtro
    anti-aliasing. No painel sem filtro, marca o alias previsto dos picos originais que
    ficam acima do novo Nyquist: f_alias = |f - round(f/fs_nova) * fs_nova|."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    rid, pos = primeiro_registro(idx, "saudável")
    if rid is None:
        return
    x = np.asarray(W[pos, 0, :], dtype=np.float64)
    A0, f0 = fft_amplitude(x, fs)
    picos0 = _picos(A0, f0, k=3, rel=0.1, fmax=fs / 2)

    fatores = FATORES_DECIMACAO[sensor]
    fig, eixos = plt.subplots(len(fatores), 3, figsize=(18, 3.8 * len(fatores)),
                              sharey=True, squeeze=False)
    for r, fator in enumerate(fatores):
        fs_nova = fs / fator
        casos = [
            (f"Original — fs = {fs:.0f} Hz", x, fs, "tab:blue"),
            (f"SEM filtro (x[::{fator}]) — fs = {fs_nova:.0f} Hz", x[::fator], fs_nova, "tab:red"),
            (f"COM filtro anti-aliasing — fs = {fs_nova:.0f} Hz",
             resample_poly(x, up=1, down=fator), fs_nova, "tab:green"),
        ]
        for c, (titulo, sinal, fs_i, cor) in enumerate(casos):
            axp = eixos[r, c]
            A, f = fft_amplitude(sinal, fs_i)
            axp.semilogy(f[1:], A[1:], lw=0.8, color=cor)
            axp.axvline(fs_nova / 2, color="k", ls="--", lw=0.8,
                        label=f"Nyquist nova = {fs_nova / 2:.0f} Hz")
            if c == 0:
                for fp, ap, _ in picos0:
                    axp.plot(fp, ap, "v", color="k", ms=5)
            if c == 1:
                for fp, _, _ in picos0:
                    if fp > fs_nova / 2:
                        fa = abs(fp - round(fp / fs_nova) * fs_nova)
                        axp.axvline(fa, color="m", ls=":", lw=1.2,
                                    label=f"alias previsto de {fp:.0f} Hz → {fa:.0f} Hz")
            axp.set_title(titulo, fontsize=9)
            axp.set_xlabel("frequência [Hz]")
            axp.grid(True, which="both", alpha=0.3)
            if c in (0, 1):
                axp.legend(fontsize=7)
        eixos[r, 0].set_ylabel(f"amplitude [{un}]")
    fig.suptitle(f"Item 7: aliasing por decimação — {sensor} ({rid}); ▼ = picos originais")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"efeito_aliasing_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_periodograma_vs_welch(W, idx, sensor, f_ruido=(2000.0, 4000.0)):
    """Periodograma (4 s, retangular) × Welch (Hann, subjanelas de 1 s, 50%).
    O CV (std/média) da PSD num trecho de ruído mostra que o periodograma não 'suaviza'
    com N maior (CV ~ 1), enquanto o Welch reduz a variância à custa de Δf."""
    fs = float(idx["fs_hz"].iloc[0])
    un = UNIT_SHORT[sensor]
    rid, pos = primeiro_registro(idx, "saudável")
    if rid is None:
        return
    x = np.asarray(W[pos, 0, :], dtype=np.float64)
    nper = int(fs)                                   # subjanelas de 1 s
    P = psd_unilateral(x, fs)
    f = np.fft.rfftfreq(x.size, 1.0 / fs)
    fw, Pw = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                   detrend=False, scaling="density")
    hi = min(f_ruido[1], fs / 2 * 0.95)
    lo = min(f_ruido[0], hi * 0.5)
    cv = lambda ff, PP: (PP[(ff >= lo) & (ff <= hi)].std()
                         / PP[(ff >= lo) & (ff <= hi)].mean())
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    eixos[0].semilogy(f[1:], P[1:], lw=0.6, color="tab:blue")
    eixos[0].set_title(f"Periodograma 4 s, retangular (Δf={f[1]:.2f} Hz)\n"
                       f"CV da PSD em {lo:.0f}–{hi:.0f} Hz = {cv(f, P):.2f}", fontsize=9)
    eixos[1].semilogy(fw[1:], Pw[1:], lw=0.9, color="tab:green")
    eixos[1].set_title(f"Welch, Hann, subjanelas de 1 s (Δf={fw[1]:.2f} Hz)\n"
                       f"CV da PSD em {lo:.0f}–{hi:.0f} Hz = {cv(fw, Pw):.2f}", fontsize=9)
    for axp in eixos:
        axp.set_xlabel("frequência [Hz]")
        axp.grid(True, which="both", alpha=0.3)
    eixos[0].set_ylabel(f"PSD [{un}²/Hz]")
    fig.suptitle(f"{sensor} ({rid}): variância do estimador espectral")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"periodograma_vs_welch_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_boxplots_espectrais(df, sensor):
    """Item 4: uma figura com todas as features; caixas por velocidade e classe.
    Potências absolutas em escala log (cauda pesada)."""
    speeds = sorted(df["speed"].unique())
    n = len(FEATS_ESPECTRAIS)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, eixos = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    eixos = np.atleast_1d(eixos).ravel()
    for axp, feat in zip(eixos, FEATS_ESPECTRAIS):
        dados, pos, cores = [], [], []
        for i, sp in enumerate(speeds):
            for j, (classe, cor) in enumerate(CORES.items()):
                m = (df["speed"] == sp) & (df["classe"] == classe)
                dados.append(df.loc[m, feat].to_numpy())
                pos.append(i * 3 + j)
                cores.append(cor)
        bp = axp.boxplot(dados, positions=pos, widths=0.8, showfliers=False,
                         patch_artist=True)
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.6)
        axp.set_xticks([i * 3 + 0.5 for i in range(len(speeds))])
        axp.set_xticklabels([f"SP{s}" for s in speeds])
        if feat.startswith("potencia_"):
            axp.set_yscale("log")
            axp.set_title(f"{feat} [{UNIT_SHORT[sensor]}²]", fontsize=9)
        else:
            axp.set_title(feat, fontsize=9)
        axp.grid(True, alpha=0.3)
    for axp in eixos[n:]:
        axp.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in CORES.values()]
    eixos[0].legend(handles, list(CORES), fontsize=8)
    fig.suptitle(f"{sensor} — features espectrais por velocidade e classe (R{REP_PAR})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"boxplots_{sensor}.png", dpi=120)
    plt.close(fig)


def auc_por_frequencia(W, idx, sensor, largura_hz=5.0, fmax=2000.0):
    """AUC (defeituoso > saudável) da potência em cada sub-banda de `largura_hz`,
    por velocidade (todas as janelas R1, canais agrupados). Mostra ONDE no eixo de
    frequência as classes se separam. EXPLORATÓRIO: usa os rótulos de todos os
    registros; para escolher bandas do classificador, refazer só com o treino."""
    fs = float(idx["fs_hz"].iloc[0])
    N = W.shape[-1]
    f = np.fft.rfftfreq(N, 1.0 / fs)
    edges = np.arange(0.0, min(fmax, fs / 2) + largura_hz / 2, largura_hz)
    ei = np.searchsorted(f, edges, side="left")
    ei = np.unique(ei)
    offs = ei[:-1] - ei[0]
    centros = 0.5 * (f[ei[:-1]] + f[ei[1:] - 1])
    rows = []
    for sp in sorted(idx["speed"].unique()):
        m = (idx["speed"] == sp).to_numpy()
        y = (idx.loc[m, "classe"] == "defeituoso").to_numpy()
        if y.all() or (~y).all():
            continue
        pos = np.where(m)[0]
        Xb = []
        for i in range(0, len(pos), CHUNK):
            P = psd_unilateral(np.asarray(W[pos[i:i + CHUNK], 0, :]), fs)
            P[:, 0] = 0.0                                   # exclui DC
            Xb.append(np.add.reduceat(P[:, ei[0]:ei[-1]], offs, axis=1))
        auc = auc_rank(np.concatenate(Xb), y)
        for c, a in zip(centros, auc):
            rows.append(dict(sensor=sensor, speed=sp, f_centro_hz=round(float(c), 2),
                             auc=float(a), auc_sep=float(max(a, 1 - a))))
    return pd.DataFrame(rows)


def plot_auc_por_frequencia(d, sensor):
    if d.empty:
        return
    fig, axp = plt.subplots(figsize=(11, 4))
    for sp, g in d.groupby("speed"):
        axp.plot(g.f_centro_hz, g.auc, lw=0.9, label=f"SP{sp}")
    axp.axhline(0.5, color="k", lw=0.6)
    axp.set_ylim(0, 1)
    axp.set_xlabel("frequência [Hz] (centro da sub-banda)")
    axp.set_ylabel("AUC (1 = defeituoso tem mais potência)")
    axp.set_title(f"{sensor}: separabilidade por sub-banda de frequência (R{REP_PAR})")
    axp.grid(True, alpha=0.3)
    axp.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"auc_por_frequencia_{sensor}.png", dpi=130)
    plt.close(fig)


def plot_dose_resposta(df, sensor, feats=("mag_dominante", "potrel_B2", "centroide")):
    """Feature × severidade (SV), uma linha por tipo de falha, por velocidade.
    Faixa cinza = saudável (mediana e IQR). Se a feature cresce/decresce de forma
    monótona com SV, isso sustenta a associação com o defeito (item 3.3.5)."""
    feats = [f for f in feats if f in df.columns]
    speeds = sorted(df["speed"].unique())
    fig, eixos = plt.subplots(len(feats), len(speeds),
                              figsize=(4.5 * len(speeds), 3.2 * len(feats)),
                              squeeze=False)
    for r, feat in enumerate(feats):
        for c, sp in enumerate(speeds):
            axp = eixos[r, c]
            h = df[(df.fault == 0) & (df.speed == sp)][feat]
            if len(h):
                q1, med, q3 = h.quantile([0.25, 0.5, 0.75])
                axp.axhspan(q1, q3, color="gray", alpha=0.25)
                axp.axhline(med, color="gray", ls="--", lw=0.8, label="saudável")
            for fl in sorted(x for x in df.fault.unique() if x != 0):
                g = df[(df.fault == fl) & (df.speed == sp)].groupby("severity")[feat]
                med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
                axp.errorbar(med.index, med.values,
                             yerr=[(med - q1).values, (q3 - med).values],
                             marker="o", capsize=3, lw=1, label=f"F{fl}")
            axp.set_xticks(sorted(df[df.fault != 0].severity.unique()))
            axp.grid(True, alpha=0.3)
            if r == 0:
                axp.set_title(f"SP{sp}")
            if c == 0:
                axp.set_ylabel(feat)
            if r == len(feats) - 1:
                axp.set_xlabel("severidade (SV)")
    eixos[0, 0].legend(fontsize=7)
    fig.suptitle(f"{sensor}: dose-resposta (mediana ± IQR, janelas R{REP_PAR})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"dose_resposta_{sensor}.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    dfs, picos, aucf = [], [], []
    for sensor in ACTIVE_SENSORS:
        if not (OUT_DIR / f"janelas_{sensor}.npy").exists():
            continue

        # Análise principal: R1 (mesma sessão)
        W, idx = carregar_sensor(sensor, repetition=REP_PAR)
        print(f"\n=== {sensor}: {len(idx)} janelas em R{REP_PAR} "
              f"({idx.record_id.nunique()} registros) ===")

        df = tabela_features_espectrais(W, idx, sensor)
        dfs.append(df)

        plot_espectros_comparativos(W, idx, sensor, canal=PAR_CANAL)   # item 2
        plot_espectros_por_falha(W, idx, sensor, canal=PAR_CANAL)      # item 2/5
        plot_efeito_duracao(W, idx, sensor)                            # item 6
        plot_efeito_duracao_zoom(W, idx, sensor)                       # item 6
        plot_efeito_aliasing(W, idx, sensor)                           # item 7
        plot_periodograma_vs_welch(W, idx, sensor)
        picos.append(tabela_picos(W, idx, sensor, canal=PAR_CANAL))    # item 5
        d_auc = auc_por_frequencia(W, idx, sensor)
        aucf.append(d_auc)
        plot_auc_por_frequencia(d_auc, sensor)
        if sensor == "ACC":
            plot_boxplots_espectrais(df, sensor)                       # item 4
            plot_dose_resposta(df, sensor)

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_DIR / "features_espectrais_longo.csv", index=False)
    pd.concat(picos, ignore_index=True).to_csv(OUT_DIR / "picos_por_condicao.csv", index=False)
    pd.concat(aucf, ignore_index=True).round(4).to_csv(OUT_DIR / "auc_por_frequencia.csv", index=False)

    resumo = full.groupby(["sensor", "eixo", "speed", "classe"])[
        FEATS_ESPECTRAIS].agg(["mean", "std"]).round(4)
    resumo.columns = [f"{f}_{est}" for f, est in resumo.columns]
    resumo.to_csv(OUT_DIR / "tabela_espectral_resumo.csv")
    print("\n=== Tabela espectral (média por classe e velocidade) ===")
    print(resumo.to_string())

    eff = effect_sizes_espectral(full, ["sensor", "eixo", "speed"])
    eff.round(3).to_csv(OUT_DIR / "tamanho_efeito_espectral.csv", index=False)
    print("\n=== Separabilidade espectral (AUC simétrica) ===")
    print(eff.pivot_table(index=["sensor", "eixo", "speed"],
                          columns="feature", values="auc_sep")
          .round(2).to_string())

    # Item 5: interpretação — frequências dominantes por condição
    print("\n=== Item 5: frequência dominante e magnitude (mediana por condição) ===")
    print(full.groupby(["sensor", "eixo", "speed", "classe"])[
        ["freq_dominante", "mag_dominante", "centroide"]].median().round(2)
          .to_string())

    # Item 6: espaçamento em frequência (Δf = fs/N)
    print("\n=== Item 6: Δf = fs/N para durações comparadas ===")
    for sensor in full["sensor"].unique():
        fs = float(full[full.sensor == sensor]["fs_hz"].iloc[0])
        for dur in DURACOES_COMPARADAS:
            N = int(fs * dur)
            print(f"  {sensor}: {dur:.1f} s -> N={N}, Δf = {fs / N:.4f} Hz")
