"""
Executa o pipeline completo, em ordem. Funciona local e no Colab.

  Local :  DRONE_DATA_DIR=/caminho/dos/mat DRONE_OUT_DIR=./saida python run_all.py
  Colab :  import os
           os.environ["DRONE_DATA_DIR"] = "/content/dados"
           os.environ["DRONE_OUT_DIR"]  = "/content/saida"
           !python run_all.py

Dependências: numpy pandas scipy matplotlib seaborn
Dataset: DronePropB (Mendeley Data, DOI 10.17632/xkvfjmm8zg.1); 108 arquivos .mat
(SP1-3, C1-3) na pasta DRONE_DATA_DIR.
"""
import subprocess
import sys
import time

ETAPAS = [
    ("3.1  seleção, qualidade e janelamento", "preprocess3_1.py"),
    ("3.2  features temporais",               "preprocess3_2.py"),
    ("3.3  FFT e features espectrais",        "preprocess3_3.py"),
    ("3.4  comparação de domínios e seleção", "preprocess3_4.py"),
    ("     teste de efeito de sessão (R1×R2×R3)", "teste_sessao.py"),
    ("     espectrogramas",                   "espectrograma.py"),
]

if __name__ == "__main__":
    for nome, script in ETAPAS:
        print(f"\n{'=' * 70}\n>>> {nome}  [{script}]\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            sys.exit(f"Falha em {script} (código {r.returncode}).")
        print(f"<<< {script} concluído em {time.time() - t0:.0f} s", flush=True)
