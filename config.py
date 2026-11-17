"""
Configuração central do pipeline (substitui os diálogos Tkinter da v1).

Caminhos: variáveis de ambiente DRONE_DATA_DIR (.mat originais) e DRONE_OUT_DIR
(saídas). No Colab:  os.environ["DRONE_DATA_DIR"] = "/content/dados"  antes de rodar.
Constantes compartilhadas por todos os scripts ficam aqui para não divergirem.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DRONE_DATA_DIR", "./dataset"))
OUT_DIR = Path(os.environ.get("DRONE_OUT_DIR", "./saida"))

# Sensores processados. IMU desligado por padrão: em R1 é quase só offset de
# orientação e triplica figuras/tabelas. Reative (ex.: pergunta da 3.6 com 3 eixos).
ACTIVE_SENSORS = ("ACC", "IMU")

AXES = {"ACC": ["ch"], "IMU": ["x", "y", "z"]}
UNITS = {"ACC": "g (assumido)", "IMU": "m/s²"}
UNIT_SHORT = {"ACC": "g", "IMU": "m/s²"}

# Análise principal: repetição 1 (mesma sessão do conjunto defeituoso).
REP_PAR = 1
PAR_SP, PAR_CANAL = 1, 1

# Features temporais
FEATS_DC = ["media", "rms", "potencia"]                    # só diagnóstico
FEATS_TEMPO_AC = ["std", "pico_abs", "pico_pico", "crista", "assimetria", "curtose"]
FEATS_TEMPO = FEATS_DC + FEATS_TEMPO_AC

# Bandas espectrais [Hz). B4 adicionada: cobre 1–5 kHz (antes só ia até 1 kHz,
# mas centroide/entropia já usavam o espectro todo). DC (f=0) sempre excluído.
BANDAS = [("B1", 0.0, 50.0), ("B2", 50.0, 300.0),
          ("B3", 300.0, 1000.0), ("B4", 1000.0, 5000.0)]

FEATS_ESPECTRAIS = (
    ["freq_dominante", "mag_dominante", "centroide", "entropia",
     "largura_banda", "planicidade", "rolloff85"]
    + [f"potencia_{n}" for n, _, _ in BANDAS]      # absoluta [unidade²]
    + [f"potrel_{n}" for n, _, _ in BANDAS]        # fração da potência total (soma = 1)
)

LIMIAR_CORR = 0.85          # |rho| >= limiar => redundante (3_4)
