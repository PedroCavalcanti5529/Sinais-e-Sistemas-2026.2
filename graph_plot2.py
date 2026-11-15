from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OUT_DIR = Path("/home/pedro/ML-Sinais/outputs")
FIG_DIR = OUT_DIR / "figuras"
AXES = {"ACC": ["ch"], "IMU": ["x", "y", "z"]}
UNITS = {"ACC": "g (assumido)", "IMU": "m/s²"}

DETALHADO = False       # True: gera também as figuras auxiliares
CANAL_RELATORIO = 1     # canal usado nas figuras por velocidade e no par do gráfico temporal
PAR_SP = 1              # velocidade do par saudável x defeituoso
PLOT_HEALTHY = None     # ex.: "F0_SV0_SP1_C1_R1"; None = escolhe o registro representativo
PLOT_FAULTY = None      # ex.: "F1_SV3_SP1_C1";    None = escolhe o registro representativo
ZOOM_S = 0.05
CHUNK = 500             # janelas processadas por vez (limita o uso de memória)

FEATS = ["media", "rms", "std", "pico_abs", "pico_pico", "crista",
         "assimetria", "curtose"]
MAIN4 = ["rms", "std", "pico_abs", "crista"]
CLASS_COLORS = {"saudável": "tab:blue", "defeituoso": "tab:red"}

# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def _bxp(a, data, labels, **kw):
    """boxplot compatível com matplotlib antigo (labels) e novo (tick_labels)."""
    try:
        return a.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return a.boxplot(data, labels=labels, **kw)


def plot_time_comparison(sensor, par):
    W = np.load(OUT_DIR / f"janelas_{sensor}.npy", mmap_mode="r")
    idx = pd.read_csv(OUT_DIR / f"janelas_{sensor}_indice.csv")
    fs = idx.fs_hz.iloc[0]
    axes = AXES[sensor]
    recs = [(par[0], "saudável"), (par[1], "defeituoso")]
    if not all((idx.record_id == r).any() for r, _ in recs):
        print(f"[AVISO] registros do gráfico ausentes em {sensor}; pulando.")
        return
    fig, ax = plt.subplots(len(axes) * 2, 2, figsize=(12, 3 * len(axes) * 2),
                           squeeze=False)
    for j, (rec, lab) in enumerate(recs):
        sel = np.where(idx.record_id == rec)[0]
        w = sel[len(sel) // 2]                      # janela central do registro
        t0 = idx.t_start_s.iloc[w]
        for a, name in enumerate(axes):
            x = np.asarray(W[w, a], dtype=np.float64)
            mu = x.mean()
            x = x - mu                              # só para visualizar (no IMU z, a gravidade esmagaria o sinal)
            t = t0 + np.arange(x.size) / fs
            n = int(ZOOM_S * fs)
            ax[2 * a, j].plot(t, x, lw=0.4, color=CLASS_COLORS[lab])
            ax[2 * a, j].set_title(f"{lab} – {rec} – eixo {name}\n"
                                   f"janela de 1 s (média {mu:.3g} removida)", fontsize=9)
            ax[2 * a + 1, j].plot(t[:n], x[:n], lw=0.8, color=CLASS_COLORS[lab])
            ax[2 * a + 1, j].set_title(f"zoom {ZOOM_S*1000:.0f} ms", fontsize=9)
            for r in (2 * a, 2 * a + 1):
                ax[r, j].set_ylabel(f"aceleração [{UNITS[sensor]}]")
                ax[r, j].set_xlabel("tempo [s]")
    for r in range(len(axes) * 2):                   # mesma escala entre condições
        lim = max(np.abs(ax[r, 0].get_ylim()).max(), np.abs(ax[r, 1].get_ylim()).max())
        for j in (0, 1):
            ax[r, j].set_ylim(-lim, lim)
    fig.suptitle(f"{sensor}: sinal no tempo, saudável × defeituoso (mesma velocidade e canal)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"tempo_{sensor}.png", dpi=150)
    plt.close(fig)


def plot_boxplots(df, sensor, detalhado=False):
    for ax_name in AXES[sensor]:
        d = df[df.eixo == ax_name]
        fig, axs = plt.subplots(1, len(MAIN4), figsize=(4 * len(MAIN4), 4))
        for a, f in zip(axs, MAIN4):
            data = [d[d.classe == c][f].values for c in CLASS_COLORS]
            bp = _bxp(a, data, list(CLASS_COLORS), patch_artist=True)
            for patch, c in zip(bp["boxes"], CLASS_COLORS.values()):
                patch.set_facecolor(c); patch.set_alpha(0.5)
            a.set_title(f)
            a.set_ylabel("adimensional" if f == "crista" else UNITS[sensor])
        fig.suptitle(f"{sensor} eixo {ax_name}: classe binária (todas as velocidades e canais)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"boxplot_classe_{sensor}_{ax_name}.png", dpi=150)
        plt.close(fig)

        if not detalhado:
            continue

        # por registro: só legível com poucos registros
        order = d.groupby("record_id").fault.first().sort_values().index
        if len(order) <= 30:
            fig, axs = plt.subplots(1, len(MAIN4), figsize=(4.5 * len(MAIN4), 4.5))
            for a, f in zip(axs, MAIN4):
                data = [d[d.record_id == r][f].values for r in order]
                bp = _bxp(a, data, list(order), patch_artist=True)
                for patch, r in zip(bp["boxes"], order):
                    patch.set_facecolor(CLASS_COLORS["saudável" if r.startswith("F0")
                                                     else "defeituoso"])
                    patch.set_alpha(0.5)
                a.set_title(f); a.tick_params(axis="x", rotation=90)
            fig.suptitle(f"{sensor} eixo {ax_name}: por registro")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"boxplot_registro_{sensor}_{ax_name}.png", dpi=150)
            plt.close(fig)
        else:
            print(f"[INFO] {len(order)} registros: gráfico por registro omitido (use por velocidade).")

        fig, axs = plt.subplots(1, len(MAIN4), figsize=(4 * len(MAIN4), 3.8))
        for a, f in zip(axs, MAIN4):
            lo, hi = np.percentile(d[f], [0.5, 99.5])
            bins = np.linspace(lo, hi, 40)
            for c, col in CLASS_COLORS.items():
                a.hist(d[d.classe == c][f], bins=bins, alpha=0.5, density=True,
                       color=col, label=c)
            a.set_title(f); a.set_ylabel("densidade")
        axs[0].legend()
        fig.suptitle(f"{sensor} eixo {ax_name}: histogramas")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"hist_{sensor}_{ax_name}.png", dpi=150)
        plt.close(fig)


def plot_by_speed(df, sensor, feats=MAIN4):
    """Boxplots por velocidade e classe, um arquivo por eixo e canal."""
    for ax_name in AXES[sensor]:
        for ch in sorted(df.channel.unique()):
            d = df[(df.eixo == ax_name) & (df.channel == ch)]
            sps = sorted(d.speed.unique())
            fig, axs = plt.subplots(1, len(feats), figsize=(4.5 * len(feats), 4))
            for a, f in zip(axs, feats):
                pos, data, cols = [], [], []
                for i, sp in enumerate(sps):
                    for j, c in enumerate(CLASS_COLORS):
                        pos.append(i * 3 + j)
                        data.append(d[(d.speed == sp) & (d.classe == c)][f].values)
                        cols.append(CLASS_COLORS[c])
                bp = a.boxplot(data, positions=pos, patch_artist=True, widths=0.8)
                for patch, col in zip(bp["boxes"], cols):
                    patch.set_facecolor(col); patch.set_alpha(0.5)
                a.set_xticks([i * 3 + 0.5 for i in range(len(sps))],
                             [f"SP{sp}" for sp in sps])
                a.set_title(f)
                a.set_ylabel("adimensional" if f == "crista" else UNITS[sensor])
            axs[0].legend(handles=[Patch(color=c, alpha=0.5, label=l)
                                   for l, c in CLASS_COLORS.items()])
            fig.suptitle(f"{sensor} eixo {ax_name}, canal C{ch}: por velocidade")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"boxplot_vel_{sensor}_{ax_name}_C{ch}.png", dpi=150)
            plt.close(fig)


def plot_corr(df, sensor):
    for ax_name in AXES[sensor]:
        C = df[df.eixo == ax_name][FEATS].corr(method="spearman")
        fig, a = plt.subplots(figsize=(7, 6))
        im = a.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        a.set_xticks(range(len(FEATS)), FEATS, rotation=90)
        a.set_yticks(range(len(FEATS)), FEATS)
        for i in range(len(FEATS)):
            for j in range(len(FEATS)):
                a.text(j, i, f"{C.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im)
        a.set_title(f"Correlação de Spearman – {sensor} eixo {ax_name}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"correlacao_{sensor}_{ax_name}.png", dpi=150)
        plt.close(fig)
        C.round(3).to_csv(OUT_DIR / f"correlacao_{sensor}_{ax_name}.csv")