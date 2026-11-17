from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import OUT_DIR, AXES, UNITS, REP_PAR, PAR_SP, PAR_CANAL as CANAL_RELATORIO


FIG_DIR = OUT_DIR / "figuras"

DETALHADO = False       # True: gera também as figuras auxiliares
PLOT_HEALTHY = None     # ex.: "F0_SV0_SP1_C1_R1"; None = escolhe o registro representativo
PLOT_FAULTY = None      # ex.: "F1_SV3_SP1_C1";    None = escolhe o registro representativo
ZOOM_S = 0.05
CHUNK = 500             # janelas processadas por vez (limita o uso de memória)

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
                                   f"janela de {x.size / fs:.0f} s (média {mu:.3g} removida)", fontsize=9)
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
        fig.suptitle(f"{sensor} eixo {ax_name}: classe binária (todas as velocidades e canais, R{REP_PAR})")
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
            fig.suptitle(f"{sensor} eixo {ax_name}, canal C{ch}: por velocidade (R{REP_PAR})")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"boxplot_vel_{sensor}_{ax_name}_C{ch}.png", dpi=150)
            plt.close(fig)


def plot_dc_effect(df, sensor):
    """Item 3.2.4: relação RMS × potência e efeito da componente média.
    (a) RMS² vs média²+std² (identidade P = mu² + sigma²);
    (b) fração da potência na componente média, mu²/P, por velocidade e classe;
    (c) RMS vs std: a distância até a diagonal é o efeito do offset."""
    for ax_name in AXES[sensor]:
        d = df[df.eixo == ax_name]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))

        lhs, rhs = d["rms"] ** 2, d["media"] ** 2 + d["std"] ** 2
        axs[0].scatter(rhs, lhs, s=4, alpha=0.4)
        lim = [0, max(lhs.max(), rhs.max())]
        axs[0].plot(lim, lim, "k--", lw=0.8, label="y = x")
        axs[0].set_xlabel(f"média² + std²  [({UNITS[sensor]})²]")
        axs[0].set_ylabel(f"RMS² = potência média  [({UNITS[sensor]})²]")
        axs[0].set_title("(a) identidade P = μ² + σ²")
        axs[0].legend()

        frac = d["media"] ** 2 / d["potencia"]
        sps = sorted(d.speed.unique())
        pos, data, cols = [], [], []
        for i, sp in enumerate(sps):
            for j, c in enumerate(CLASS_COLORS):
                pos.append(i * 3 + j)
                data.append(frac[(d.speed == sp) & (d.classe == c)].values)
                cols.append(CLASS_COLORS[c])
        bp = axs[1].boxplot(data, positions=pos, patch_artist=True, widths=0.8,
                            showfliers=False)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        axs[1].set_xticks([i * 3 + 0.5 for i in range(len(sps))],
                          [f"SP{sp}" for sp in sps])
        axs[1].set_ylabel("μ² / P  (fração da potência no DC)")
        axs[1].set_title("(b) peso da componente média")
        axs[1].legend(handles=[Patch(color=c, alpha=0.5, label=l)
                               for l, c in CLASS_COLORS.items()])

        for c, col in CLASS_COLORS.items():
            q = d[d.classe == c]
            axs[2].scatter(q["std"], q["rms"], s=4, alpha=0.4, color=col, label=c)
        lim = [0, max(d["rms"].max(), d["std"].max())]
        axs[2].plot(lim, lim, "k--", lw=0.8)
        axs[2].set_xlabel(f"std [{UNITS[sensor]}]")
        axs[2].set_ylabel(f"RMS [{UNITS[sensor]}]")
        axs[2].set_title("(c) RMS × std (acima da diagonal = offset)")
        axs[2].legend()

        fig.suptitle(f"{sensor} eixo {ax_name}: RMS, potência e componente média (R{REP_PAR})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"efeito_dc_{sensor}_{ax_name}.png", dpi=150)
        plt.close(fig)
