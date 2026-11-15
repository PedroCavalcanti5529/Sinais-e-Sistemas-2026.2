"""
Parte 3.3 - Análise espectral (FFT) dos sinais de vibração  (v2)

Entrada : janelas_{ACC,IMU}.npy + janelas_{...}_indice.csv + registros_metadados.csv
          (janelas de 4 s da 3.1, 50% de overlap)
Saídas  : features_espectrais_longo.csv      (uma linha por janela × eixo)
          tabela_espectral_resumo.csv        (média/desvio por classe e velocidade)
          tamanho_efeito_espectral.csv       (AUC por feature/sensor/eixo/velocidade)
          figuras_espectrais/espectros_*.png        (item 2)
          figuras_espectrais/boxplots_ACC.png       (item 4)
          figuras_espectrais/efeito_duracao_*.png   (item 6: Δf muda com N)
          figuras_espectrais/efeito_aliasing_*.png  (item 7: decimação e Nyquist)

Normalização adotada
  - Amplitude (pico da senoide equivalente): A[k] = 2 * |X[k]| / N  (k > 0)
    A[0] = |X[0]| / N  (DC, sem fator 2)
  - PSD unilateral: P[k] = 2 * |X[k]|^2 / (fs * N)  (k > 0), P[0] = |X[0]|^2 / (fs*N)
    Integrar P[k] * Δf em uma banda dá a potência média (unidade^2) na banda.

Bandas (justificativa) — o bin de DC (f = 0) é EXCLUÍDO de todas as bandas,
porque carrega o offset da sessão (mesmo confundimento de media/rms na 3.2).
  - B1 = (0, 50] Hz : movimentos estruturais/rolagem; abaixo da dinâmica de hélice.
  - B2 = 50–300 Hz  : dinâmica de hélice (rotação x nº de pás). É onde a falha
                      tende a aparecer primeiro (desequilíbrio, dano em pá).
  - B3 = 300–1000 Hz: harmônicos e ressonâncias de alta ordem (limitada por Nyquist).
  A escolha é relativa e conservadora porque a rotação do drone não está
  documentada. Se a rotação for conhecida, as bandas devem ser redefinidas em
  torno de f_rot e seus harmônicos.

Notas de confundimento (herdadas da 3.2)
  - R2 e R3 são de outra sessão; a análise principal usa repetition == 1 (R1).
  - O ACC é o sensor útil; o IMU em R1 é quase todo offset de orientação.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/home/pedro/ML-Sinais/outputs")
FIG_DIR = OUT_DIR / "figuras_espectrais"

AXES = {"ACC": ["ch"], "IMU": ["x", "y", "z"]}
REP_PAR = 1
PAR_SP, PAR_CANAL = 1, 1            # registros usados nas figuras dos itens 6 e 7
CORES = {"saudável": "tab:blue", "defeituoso": "tab:red"}

# Bandas em Hz. Fixas porque a rotação do drone não está documentada.
BANDAS = [
    ("B1", 0.0, 50.0),
    ("B2", 50.0, 300.0),
    ("B3", 300.0, 1000.0),
]

# Item 6 (efeito da duração): recorta 1 s da janela de 4 s
DURACOES_COMPARADAS = (1.0, 4.0)
# Item 7 (aliasing): reduzir a fs em um fator inteiro
FATOR_DECIMACAO = {"ACC": 5, "IMU": 2}   # 10 kHz -> 2 kHz; 1 kHz -> 500 Hz

FEATS_ESPECTRAIS = (
    ["freq_dominante", "mag_dominante", "centroide", "entropia", "largura_banda"]
    + [f"potencia_{nome}" for nome, _, _ in BANDAS]
)


# ----------------------------------------------------------------------------
# Carregamento (mesmo contrato da 3.2)
# ----------------------------------------------------------------------------
def carregar_sensor(sensor, repetition=None):
    """Retorna (W, meta_por_janela). W: (n_janelas, n_eixos, n_amostras)."""
    W = np.load(OUT_DIR / f"janelas_{sensor}.npy", mmap_mode="r")
    idx = pd.read_csv(OUT_DIR / f"janelas_{sensor}_indice.csv")
    meta = pd.read_csv(OUT_DIR / "registros_metadados.csv")
    meta = meta[meta.sensor == sensor][
        ["record_id", "fault", "severity", "speed", "channel", "repetition"]
    ].drop_duplicates("record_id")
    idx = idx.merge(meta, on="record_id", how="left")
    idx["classe"] = np.where(idx["fault"] == 0, "saudável", "defeituoso")
    if repetition is not None:
        mask = (idx["repetition"] == repetition).to_numpy()
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


def features_espectrais_janela(W_eixo, fs):
    """W_eixo: (n_janelas, n_amostras) -> dict de arrays (n_janelas,)."""
    A, f = fft_amplitude(W_eixo, fs)          # (n_janelas, n_freq)
    P = psd_unilateral(W_eixo, fs)
    df = f[1] - f[0]

    # Todas as medidas ignoram o DC para não pegar o offset
    A_sem_dc = A[:, 1:]
    f_sem_dc = f[1:]
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

    out = {
        "freq_dominante": freq_dom,
        "mag_dominante": mag_dom,
        "centroide": centroide,
        "entropia": entropia,
        "largura_banda": largura,
    }
    # Potência em bandas (integração da PSD, unidade^2), sem o bin de DC
    for nome, lo, hi in BANDAS:
        hi_eff = min(hi, fs / 2.0)        # respeita Nyquist
        mask = (f > 0) & (f >= lo) & (f < hi_eff)
        out[f"potencia_{nome}"] = P[:, mask].sum(axis=1) * df
    return out


def tabela_features_espectrais(W, idx, sensor):
    """W: (n_janelas, n_eixos, n_amostras) -> DataFrame longo."""
    fs = float(idx["fs_hz"].iloc[0]) if len(idx) else 1.0
    partes = []
    for a, ax in enumerate(AXES[sensor]):
        F = features_espectrais_janela(np.asarray(W[:, a, :]), fs)
        d = idx.copy()
        d["sensor"] = sensor
        d["eixo"] = ax
        for k, v in F.items():
            d[k] = v
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------------
# Tabelas de efeito
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


# ----------------------------------------------------------------------------
# Figuras (itens 2, 4, 6, 7)
# ----------------------------------------------------------------------------
def plot_espectros_comparativos(W, idx, sensor, canal=None):
    """Item 2: amplitude média por classe e velocidade, por eixo."""
    fs = float(idx["fs_hz"].iloc[0])
    for a, ax in enumerate(AXES[sensor]):
        fig, eixos = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for j, sp in enumerate(sorted(idx["speed"].unique())):
            axp = eixos[j]
            for classe, cor in CORES.items():
                m = (idx["speed"] == sp) & (idx["classe"] == classe)
                if canal is not None:
                    m &= (idx["channel"] == canal)
                if not m.any():
                    continue
                A, f = fft_amplitude(np.asarray(W[m.to_numpy(), a, :]), fs)
                axp.semilogy(f[1:], A[:, 1:].mean(axis=0), color=cor,
                             label=classe, lw=1)
            axp.set_title(f"{sensor}/{ax} — SP{sp}")
            axp.set_xlabel("frequência [Hz]")
            axp.set_xscale("log")
            axp.grid(True, which="both", alpha=0.3)
            if j == 0:
                axp.set_ylabel("amplitude média")
        eixos[0].legend()
        sufixo = f"_C{canal}" if canal is not None else ""
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"espectros_{sensor}_{ax}{sufixo}.png", dpi=120)
        plt.close(fig)


def plot_efeito_duracao(W, idx, sensor):
    """Item 6: mesma janela (4 s) analisada inteira e recortada em 1 s.
    Linhas = classe, colunas = duração. Δf = fs/N."""
    fs = float(idx["fs_hz"].iloc[0])
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
                axp.set_ylabel("amplitude")
    fig.suptitle(f"Item 6: efeito da duração — {sensor}/{AXES[sensor][0]}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"efeito_duracao_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_efeito_aliasing(W, idx, sensor):
    """Item 7: decimação por fator inteiro SEM filtro (aliasing real) e COM
    filtro anti-aliasing (contraste)."""
    fs = float(idx["fs_hz"].iloc[0])
    fator = FATOR_DECIMACAO[sensor]
    fs_nova = fs / fator

    rid, pos = primeiro_registro(idx, "saudável")
    if rid is None:
        return
    x = np.asarray(W[pos, 0, :], dtype=np.float64)

    casos = [
        (f"Original — fs = {fs:.0f} Hz", x, fs, "tab:blue"),
        (f"Decimado SEM filtro (x[::{fator}]) — fs = {fs_nova:.0f} Hz",
         x[::fator], fs_nova, "tab:red"),
        (f"Decimado COM filtro anti-aliasing — fs = {fs_nova:.0f} Hz",
         resample_poly(x, up=1, down=fator), fs_nova, "tab:green"),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    for axp, (titulo, sinal, fs_i, cor) in zip(eixos, casos):
        A, f = fft_amplitude(sinal, fs_i)
        axp.semilogy(f[1:], A[1:], lw=0.8, color=cor)
        axp.axvline(fs_nova / 2, color="k", ls="--", lw=0.8,
                    label=f"Nyquist nova = {fs_nova/2:.0f} Hz")
        axp.set_title(titulo, fontsize=9)
        axp.set_xlabel("frequência [Hz]")
        axp.grid(True, which="both", alpha=0.3)
    eixos[0].set_ylabel("amplitude")
    eixos[0].legend()
    fig.suptitle(f"Item 7: aliasing por decimação (fator {fator}) — "
                 f"{sensor} ({rid})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"efeito_aliasing_{sensor}.png", dpi=120)
    plt.close(fig)


def plot_boxplots_espectrais(df, sensor):
    """Item 4: uma única figura com as 8 features; caixas por velocidade e classe."""
    speeds = sorted(df["speed"].unique())
    fig, eixos = plt.subplots(2, 4, figsize=(18, 7))
    for axp, feat in zip(eixos.ravel(), FEATS_ESPECTRAIS):
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
        axp.set_title(feat)
        axp.grid(True, alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6)
               for c in CORES.values()]
    eixos[0, 0].legend(handles, list(CORES), fontsize=8)
    fig.suptitle(f"{sensor} — features espectrais por velocidade e classe (R{REP_PAR})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"boxplots_{sensor}.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for sensor in AXES:
        if not (OUT_DIR / f"janelas_{sensor}.npy").exists():
            continue

        # Análise principal: R1 (mesma sessão)
        W, idx = carregar_sensor(sensor, repetition=REP_PAR)
        print(f"\n=== {sensor}: {len(idx)} janelas em R{REP_PAR} "
              f"({idx.record_id.nunique()} registros) ===")

        df = tabela_features_espectrais(W, idx, sensor)
        dfs.append(df)

        plot_espectros_comparativos(W, idx, sensor, canal=PAR_CANAL)   # item 2
        plot_efeito_duracao(W, idx, sensor)                            # item 6
        plot_efeito_aliasing(W, idx, sensor)                           # item 7
        if sensor == "ACC":
            plot_boxplots_espectrais(df, sensor)                       # item 4

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_DIR / "features_espectrais_longo.csv", index=False)

    resumo = full.groupby(["sensor", "eixo", "speed", "classe"])[
        FEATS_ESPECTRAIS].agg(["mean", "std"]).round(4)
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
            print(f"  {sensor}: {dur:.1f} s -> N={N}, Δf = {fs/N:.4f} Hz")