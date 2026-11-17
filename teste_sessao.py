"""
Teste de efeito de SESSÃO: saudável R1 × R2 × R3 (ACC).

Pergunta: as features AC (temporais) e espectrais — as que entram no ML — separam
repetições do MESMO estado saudável? Se não (AUC ≈ 0,5), R2/R3 podem ser adicionados
ao conjunto saudável (6 → 18 registros, alivia o desbalanceamento). Se separam, a
sessão afeta até as features AC e a análise deve ficar restrita a R1; o resultado
vira candidato à pergunta da 3.6 ("sessão importa mais que falha?").

Comparação: mesma velocidade e canal, uma repetição contra outra. Cada lado é UM
registro (janelas do mesmo registro não são independentes): indicativo, não teste.
As features DC (media/rms/potencia) entram só como CONTROLE POSITIVO (esperado alto).
Piso de ruído: com poucas janelas por registro, a AUC entre duas amostras da MESMA
distribuição oscila (≈ 0,5–0,7). Por isso o limiar de decisão é 0,65 sobre a MEDIANA de
todas as features/velocidades/canais, e não sobre uma feature isolada.

Saídas:
  teste_sessao_auc.csv     AUC por par de repetições × velocidade × canal × feature
  teste_sessao_resumo.csv  mediana da AUC simétrica por grupo de feature e par
  teste_sessao_datas.csv   data de aquisição por repetição (AccData.DATE)
"""
import numpy as np
import pandas as pd

from config import OUT_DIR, FEATS_DC, FEATS_TEMPO_AC, FEATS_ESPECTRAIS
from preprocess3_2 import extract_features, CHUNK
from preprocess3_3 import carregar_sensor, tabela_features_espectrais
from utils import auc_rank

PARES = [(1, 2), (1, 3), (2, 3)]
LIMIAR_SESSAO = 0.65        # mediana de AUC simétrica acima disso = sessão separável


def tabela_saudavel():
    """Features temporais + espectrais das janelas saudáveis de TODAS as repetições."""
    W, idx = carregar_sensor("ACC", repetition=None, so_saudavel=True)
    blocos = [extract_features(W[i:i + CHUNK]) for i in range(0, len(W), CHUNK)]
    F = {k: np.concatenate([b[k] for b in blocos])[:, 0] for k in blocos[0]}
    d = idx.copy()
    for k in FEATS_DC + FEATS_TEMPO_AC:
        d[k] = F[k]
    esp = tabela_features_espectrais(W, idx, "ACC")
    for k in FEATS_ESPECTRAIS:
        d[k] = esp[k].to_numpy()
    return d


def comparar(d):
    grupos = {**{f: "DC (controle)" for f in FEATS_DC},
              **{f: "AC temporal" for f in FEATS_TEMPO_AC},
              **{f: "espectral" for f in FEATS_ESPECTRAIS}}
    feats = list(grupos)
    rows = []
    for (sp, ch), g in d.groupby(["speed", "channel"]):
        for a, b in PARES:
            ga, gb = g[g.repetition == a], g[g.repetition == b]
            if ga.empty or gb.empty:
                continue
            X = np.vstack([ga[feats].to_numpy(), gb[feats].to_numpy()])
            y = np.r_[np.ones(len(ga), bool), np.zeros(len(gb), bool)]
            auc = auc_rank(X, y)
            for f, v in zip(feats, auc):
                rows.append(dict(par=f"R{a}×R{b}", speed=sp, channel=ch, feature=f,
                                 grupo=grupos[f], auc=v, auc_sep=max(v, 1 - v)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = tabela_saudavel()
    print(f"{len(d)} janelas saudáveis; registros por repetição:")
    print(d.groupby("repetition").record_id.nunique().to_string())

    meta = pd.read_csv(OUT_DIR / "registros_metadados.csv")
    datas = (meta[(meta.sensor == "ACC") & (meta.fault == 0)]
             .groupby("repetition").date.agg(lambda s: " | ".join(sorted(set(s)))))
    datas.to_csv(OUT_DIR / "teste_sessao_datas.csv")
    print("\nDatas de aquisição (saudável) por repetição:")
    print(datas.to_string())

    r = comparar(d)
    r.round(4).to_csv(OUT_DIR / "teste_sessao_auc.csv", index=False)
    resumo = (r.groupby(["par", "grupo"]).auc_sep.median().unstack().round(3))
    resumo.to_csv(OUT_DIR / "teste_sessao_resumo.csv")
    print("\n=== Mediana da AUC simétrica entre repetições saudáveis ===")
    print(resumo.to_string())

    pior = r[r.grupo != "DC (controle)"].groupby("feature").auc_sep.median() \
            .sort_values(ascending=False).head(5)
    print("\nFeatures AC/espectrais mais sensíveis à sessão (mediana):")
    print(pior.round(3).to_string())

    med = r[r.grupo != "DC (controle)"].auc_sep.median()
    print(f"\nMediana geral (AC + espectral) = {med:.2f}")
    if med < LIMIAR_SESSAO:
        print("→ Sessão NÃO separa de forma consistente as features AC/espectrais: "
              "R2/R3 podem ser testados como reforço do conjunto saudável.")
    else:
        print("→ Sessão separa mesmo as features AC/espectrais: manter a análise em R1 "
              "e tratar o efeito de sessão como pergunta da 3.6.")
    print("(Confirme também as datas acima: repetições em dias distintos reforçam o "
          "efeito de sessão.)")
