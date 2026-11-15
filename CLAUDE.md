# CLAUDE.md — Pipeline de vibração de drones

## Contexto
Análise de sinais de acelerômetro (ACC, 10 kHz) e IMU (1 kHz) de um drone
com hélices saudáveis vs. defeituosas. Dataset: DronePropB Ground Testing.
Objetivo final: classificar condição da hélice (saudável vs. falha) e
severidade.

## Estrutura do projeto
- `preprocess3-1.py` — seleção, dedup, verificação de qualidade, janelamento
- `preprocess3-2.py` — extração de features no domínio do tempo
- `graph_plot2.py` — funções de plot (consumidas pela 3.2 quando `PLOT=True`)
- `outputs/` — CSVs, NPYs e figuras (gerados pelas duas etapas)
- `.venv/` — ambiente Python (pandas, numpy, scipy, matplotlib)

## Convenções do dataset
- Nome: `F{falha}_SV{severidade}_SP{velocidade}_C{canal}[_R{repetição}].mat`
- `F=0` saudável; `SP1-3` e `C1-3` selecionados (SP0/C4 descartados)
- **R1 = set/2024** (única sessão com saudável **e** falha)
- **R2/R3 = nov/2024** (só saudável, outra sessão) → **ruído de sessão**
- Duplicados: C2 saudável é cópia exata de C1 → dedup por hash do ACC
- Total: 108 arquivos → 99 processados (9 duplicados ignorados)

## Contrato entre 3.1 e 3.2
- 3.1 salva `janelas_{ACC,IMU}.npy` + `janelas_{...}_indice.csv`
  (índice **enxuto**: `record_id, window_idx, start_sample, t_start_s, fs_hz, win_samples`)
- 3.1 salva `registros_metadados.csv` com metadados por registro
  (`fault, severity, speed, channel, repetition, date, fs_hz, n_samples`)
- 3.2 faz **merge** do índice com `registros_metadados.csv` por `record_id`
  para recuperar `fault/speed/channel/repetition` — **não remover esse merge**,
  ele é o que evita `KeyError: 'fault'`.
- 3.1 deve rodar com `REMOVE_DC = False` (3.2 precisa da média/DC para diagnóstico).

## Decisões já tomadas (não reverter sem evidência)
- **`AUTO_TRIM = False`** — o trim não melhora a separação; rodar sem ele.
- **`REMOVE_DC = False`** — `media` é feature de **diagnóstico** (revela
  confundimento de sessão). Todas as features AC já são invariantes a DC.
- **`media`/`rms` não são features de ML** — em R1 dão AUC ~0.52 (ruído).
  Usá-las só para auditoria (a AUC "alta" agregada é sessão, não falha).
- **Filtrar `repetition == 1`** para treino/análise principal.
- **Treinar por velocidade** (SP1, SP2, SP3 separados). Agregar velocidades
  destrói o sinal (SP3 domina o `std` por ser 10× SP1).
- **`diagnostico_sessao` foi removido** da 3.2 (redundante com a análise R1).
- **Formato longo é o canônico** (`features_tempo_longo.csv`); formato largo
  foi removido.

## Achados empíricos (estado atual)
- **ACC separa** em R1: SP1 (`rms`, `assimetria` ~0.72), SP3 (`assimetria`
  0.80, `curtose` 0.78, `crista` 0.76). SP2 é fraco (~0.62).
- **IMU em R1 é quase todo artefato**: `media` = offset de orientação, não
  vibração. Exceção suspeita: `IMU/x rms` em SP2 (~0.77) — validar visualmente
  antes de confiar.
- **AUC agregada engana**: `IMU/z media` 0.86 (todas) → 0.53 (R1).
  Sempre liderar com tabela R1 × velocidade.

## Comandos
```bash
.venv/bin/python preprocess3-1.py   # gera janelas + metadados
.venv/bin/python preprocess3-2.py   # gera features + tabelas de efeito