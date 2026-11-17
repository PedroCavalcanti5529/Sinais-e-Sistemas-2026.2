"""
Seleção, caracterização e preparação dos dados de vibração de drones (v3).

Convenção: F{falha}_SV{severidade}_SP{velocidade}_C{canal}[_R{repetição}].mat
  F=0 saudável; SP1-3 e C1-3 selecionados (SP0 e C4 descartados).
  ACC = acelerômetro 10 kHz (1 eixo); IMU = 3 eixos (opcional).
  C2 saudável é cópia exata de C1 → deduplicado via hash do ACC.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# Caminhos e sensores ativos vêm do config.py (sem diálogos; roda no Colab).
from config import DATA_DIR, OUT_DIR, ACTIVE_SENSORS

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
FNAME_RE = re.compile(r"F(\d+)_SV(\d+)_SP(\d+)_C(\d+)(?:_R(\d+))?", re.IGNORECASE)
SPEEDS = (1, 2, 3)
CHANNELS = (1, 2, 3)
EXPECTED_N_FILES = 108  # 27 saudáveis + 81 com falha

# sensor -> (variável no .mat, nomes dos eixos, unidade)
SENSOR_SPEC = {
    "ACC": ("AccData", ["ch"], "g"),
    "IMU": ("IMUData", ["x", "y", "z"], "m/s^2"),
}

WINDOW_S = 4.0
OVERLAP = 0.5
AUTO_TRIM = False
TRIM_BLOCK_S = 0.25
TRIM_THRESH = 0.6
REMOVE_DC = False


def _keep(p: Path) -> bool:
    m = FNAME_RE.search(p.stem)
    if not m:
        return False
    _, _, sp, c, _ = m.groups()
    return int(sp) in SPEEDS and int(c) in CHANNELS


def list_selected_files(data_dir=DATA_DIR, expected=EXPECTED_N_FILES):
    """Arquivos .mat selecionados (SP1-3, C1-3). Falha com mensagem clara se o
    diretório não existe ou a contagem diverge (antes era um assert no import)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"DATA_DIR não encontrado: {data_dir.resolve()}. Defina a variável "
            "de ambiente DRONE_DATA_DIR com a pasta dos .mat.")
    files = sorted(p.name for p in data_dir.glob("*.mat") if _keep(p))
    if expected is not None and len(files) != expected:
        raise AssertionError(f"{len(files)} arquivos; esperado {expected}.")
    return files


# ----------------------------------------------------------------------------
# 1) Leitura e metadados
# ----------------------------------------------------------------------------
def parse_filename(path: Path) -> dict:
    m = FNAME_RE.search(path.stem)
    if not m:
        raise ValueError(f"Nome fora do padrão: {path.name}")
    f, sv, sp, c, r = m.groups()
    return dict(fault=int(f), severity=int(sv), speed=int(sp), channel=int(c),
                repetition=int(r) if r else 1)


def load_mat(path: Path) -> dict:
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def get_sensor(mat: dict, sensor: str):
    """Retorna (X, t, fs) com X em (n_eixos, n_amostras), ou None se ausente."""
    var = SENSOR_SPEC[sensor][0]
    if var not in mat:
        return None
    s = mat[var]
    X = np.asarray(s.Vib_Data, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    elif X.shape[0] > X.shape[1]:
        X = X.T
    t = np.asarray(s.t, dtype=np.float64).ravel()
    return X, t, float(s.samplingRate)


def record_metadata(path: Path, mat: dict) -> list[dict]:
    """Uma linha por (registro, sensor) presente no arquivo."""
    base = parse_filename(path)
    acc_date = str(getattr(mat.get("AccData"), "DATE", ""))
    rows = []
    for sensor in ACTIVE_SENSORS:
        got = get_sensor(mat, sensor)
        if got is None:
            continue
        X, _, fs = got
        unit = SENSOR_SPEC[sensor][2]
        rows.append(dict(
            record_id=path.stem, file=path.name, sensor=sensor, **base,
            unit=unit, date=acc_date, fs_hz=fs, n_samples=X.shape[1],
            duration_s=X.shape[1] / fs, axes=",".join(SENSOR_SPEC[sensor][1]),
        ))
    return rows


# ----------------------------------------------------------------------------
# 2) Verificação de qualidade (por eixo)
# ----------------------------------------------------------------------------
def check_axis(x, t, fs, block_s=5.0) -> dict:
    dt = np.diff(t)
    nominal = 1 / fs
    k = int(fs)
    n = int(block_s * fs)
    sd = x[: (x.size // n) * n].reshape(-1, n).std(axis=1)

    gaps = dt > 1.5 * nominal
    lost = int(np.round((dt[gaps] / nominal - 1).sum()))

    q = {
        "n_nan_inf": int((~np.isfinite(x)).sum()),
        "n_lacunas": int(gaps.sum()),
        "amostras_perdidas": lost,
        "tempo_consistente_com_fs": bool(
            lost / x.size < 1e-3 and (dt > 0.5 * nominal).all()),
        "t_monotonico": bool((dt > 0).all()),
        "sinal_constante": bool(np.ptp(x) == 0),
        "frac_nos_extremos": float(((x == x.max()) | (x == x.min())).mean()),
        "offset_dc": float(x.mean()),
        "std": float(x.std()),
        "cv_std_blocos": float(sd.std() / sd.mean()),
        "razao_std_ini_fim": float(x[:k].std() / x[-k:].std()),
    }
    q["transitorio_suspeito"] = bool(
        q["cv_std_blocos"] > 0.10 or not (0.7 < q["razao_std_ini_fim"] < 1.4))
    q["aprovado"] = bool(
        q["n_nan_inf"] == 0 and q["tempo_consistente_com_fs"]
        and q["t_monotonico"] and not q["sinal_constante"]
        and q["frac_nos_extremos"] < 1e-3)
    return q


# ----------------------------------------------------------------------------
# 3) Pré-processamento
# ----------------------------------------------------------------------------
def find_steady_state(X, fs, block_s=TRIM_BLOCK_S, thresh=TRIM_THRESH):
    """(i0, i1): primeiro/último bloco em regime. Usa std (imune a DC)."""
    n = int(block_s * fs)
    nb = X.shape[1] // n
    rms = X[:, : nb * n].reshape(X.shape[0], nb, n).std(axis=2)
    rel = (rms / np.median(rms, axis=1, keepdims=True)).mean(axis=0)
    ok = np.where(rel >= thresh)[0]
    return int(ok[0] * n), int((ok[-1] + 1) * n)


def preprocess(X, fs, auto_trim=AUTO_TRIM, remove_dc=REMOVE_DC):
    """Recorta transitórios e (opcionalmente) remove DC. Sem filtragem."""
    i0, i1 = (find_steady_state(X, fs) if auto_trim else (0, X.shape[1]))
    Y = X[:, i0:i1].copy()
    if remove_dc:
        Y -= Y.mean(axis=1, keepdims=True)
    return Y, i0 / fs, i0, i1


# ----------------------------------------------------------------------------
# 4) Janelamento
# ----------------------------------------------------------------------------
def find_gaps(t, fs):
    """Índice da primeira amostra após cada lacuna."""
    return np.where(np.diff(t) > 1.5 / fs)[0] + 1


def make_windows(Y, fs, meta, offset_s=0.0, gaps=(), win_s=WINDOW_S,
                 overlap=OVERLAP):
    """Retorna (W, idx, n_descartadas). W: (n_janelas, n_eixos, win)."""
    win = int(round(win_s * fs))
    step = max(1, int(round(win * (1 - overlap))))
    starts = np.arange(0, Y.shape[1] - win + 1, step)

    n_before = len(starts)
    for g in np.asarray(gaps, dtype=int):
        starts = starts[~((starts < g) & (g < starts + win))]
    n_dropped = n_before - len(starts)

    W = np.stack([Y[:, s: s + win] for s in starts]).astype(np.float32)
    idx = pd.DataFrame({
        "record_id": meta["record_id"],
        "window_idx": np.arange(len(starts)),
        "start_sample": starts,
        "t_start_s": offset_s + starts / fs,
        "fs_hz": fs,
        "win_samples": win,
    })
    return W, idx, n_dropped


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def build_dataset(files, data_dir=DATA_DIR):
    metas, checks = [], []
    W_by, idx_by = {s: [] for s in ACTIVE_SENSORS}, {s: [] for s in ACTIVE_SENSORS}
    seen, dups = {}, []

    for fn in files:
        path = Path(data_dir) / fn
        mat = load_mat(path)

        acc = get_sensor(mat, "ACC")
        if acc is not None:
            h = hashlib.md5(acc[0].tobytes()).hexdigest()
            if h in seen:
                dups.append(dict(arquivo=fn, igual_a=seen[h]))
                continue
            seen[h] = fn

        for meta in record_metadata(path, mat):
            sensor = meta["sensor"]
            X, t, fs = get_sensor(mat, sensor)
            axes = SENSOR_SPEC[sensor][1]

            qs = [{"record_id": meta["record_id"], "sensor": sensor,
                   "axis": axes[i], **check_axis(X[i], t, fs)}
                  for i in range(X.shape[0])]
            checks.extend(qs)
            if not all(q["aprovado"] for q in qs):
                continue

            Y, off_s, i0, i1 = preprocess(X, fs)
            meta.update(trim_start_s=i0 / fs, trim_end_s=(X.shape[1] - i1) / fs)
            gaps = find_gaps(t, fs) - i0
            W, idx, n_drop = make_windows(Y, fs, meta, offset_s=off_s, gaps=gaps)
            meta["janelas_descartadas_lacuna"] = n_drop

            metas.append(meta)
            W_by[sensor].append(W)
            idx_by[sensor].append(idx)

    df_meta = pd.DataFrame(metas)
    df_check = pd.DataFrame(checks)
    W = {s: np.concatenate(v) for s, v in W_by.items() if v}
    idx = {s: pd.concat(v, ignore_index=True) for s, v in idx_by.items() if v}
    return (df_meta, df_check, W, idx,
            pd.DataFrame(dups, columns=["arquivo", "igual_a"]))


if __name__ == "__main__":
    SELECTED_FILES = list_selected_files()
    print(f"{len(SELECTED_FILES)} arquivos selecionados "
          f"(SP{list(SPEEDS)}, C{list(CHANNELS)}).")

    df_meta, df_check, W, idx, dups = build_dataset(SELECTED_FILES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in W:
        np.save(OUT_DIR / f"janelas_{s}.npy", W[s])
        idx[s].to_csv(OUT_DIR / f"janelas_{s}_indice.csv", index=False)
    df_meta.to_csv(OUT_DIR / "registros_metadados.csv", index=False)
    df_check.to_csv(OUT_DIR / "verificacao_qualidade.csv", index=False)
    dups.to_csv(OUT_DIR / "registros_duplicados.csv", index=False)

    print(f"Duplicados ignorados: {len(dups)} (esperado 9). "
          f"Processados: {df_meta.record_id.nunique()} registros.")

    pd.set_option("display.width", 220, "display.max_columns", 50)
    print("\n=== Resumo ===")
    print(df_meta.groupby(["sensor", "fault"]).size().unstack(fill_value=0)
          .to_string())

    aprov = df_check.groupby(["record_id", "sensor"]).aprovado.all()
    reprov = list(aprov[~aprov].index)
    transit = int(df_check.groupby(["record_id", "sensor"])
                  .transitorio_suspeito.any().sum())
    print(f"\nReprovados: {reprov or 'nenhum'}")
    print(f"Transitório suspeito: {transit}")
    print(f"Janelas descartadas por lacuna: "
          f"{int(df_meta.janelas_descartadas_lacuna.sum())}")

    print("\nDuração usada [s] por sensor e velocidade (derivada de "
          "trim_start/trim_end):")
    print((df_meta.n_samples / df_meta.fs_hz
           - df_meta.trim_start_s - df_meta.trim_end_s)
          .groupby([df_meta.sensor, df_meta.speed])
          .agg(["min", "median", "max"]).round(1).to_string())

    print("\n=== Janelamento ===")
    for s in W:
        print(f"{s}: {WINDOW_S}s ({idx[s].win_samples.iloc[0]} amostras @ "
              f"{idx[s].fs_hz.iloc[0]:.0f} Hz), overlap={OVERLAP:.0%}, "
              f"W={W[s].shape} {W[s].dtype}")