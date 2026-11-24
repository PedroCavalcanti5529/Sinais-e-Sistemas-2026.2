# Análise de Vibração de Hélices de Drone (DronePropB) — CIN0020

Repositório do projeto da disciplina **CIN0020 — Sinais e Sistemas (2026.2)** do Centro de Informática (CIn/UFPE). O projeto aplica técnicas de processamento de sinais nos domínios do tempo e da frequência sobre dados reais de aceleração do dataset **DronePropB** para caracterizar e diferenciar hélices saudáveis e defeituosas.

> **Status atual:** 1ª Entrega concluída (Seções 3.1, 3.2, 3.3, 3.4 e 3.6 em notebook único e linear). A etapa de classificação por *Machine Learning* (Seção 3.5) será incorporada na 2ª Entrega.

---

## Estrutura do Repositório

```text
.
├── projeto_simplificado.ipynb   # Notebook único e linear (Seções 3.1 a 3.4 e 3.6)
├── requirements.txt             # Dependências Python do projeto
└── README.md                    # Instruções de uso e documentação
```

---

## Escopo dos Dados

Os dados utilizados pertencem ao **DronePropB — Ground Testing Dataset for Commercial Drones with Defective Propellers** (Mendeley Data, DOI: `10.17632/xkvfjmm8zg.1`).

Recorte adotado no notebook:
* **Sensor:** acelerômetro piezoelétrico `AccData` (`ACC`, 1 eixo, $f_s = 10\,000\text{ Hz}$, unidade assumida como $g$).
* **Condições selecionadas:** velocidades `SP1`, `SP2` e `SP3`, canais `C1`, `C2` e `C3`, repetição `R1` (sessão de set/2024).
* **Deduplicação:** o canal `C2` da condição saudável (`F0`) é cópia idêntica do `C1` e é removido automaticamente.
* **Total processado:** **87 registros** (6 saudáveis + 81 defeituosos), segmentados em **1.923 janelas** de **4 s** ($40\,000$ amostras) com **50% de sobreposição**.

---

## Dependências

* Python 3.10+
* Bibliotecas: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn` (usada para `roc_auc_score`) e `jupyter` (para execução local).

> **Observação:** caso `scikit-learn` ainda não esteja listado em `requirements.txt`, instale-o juntamente com as demais dependências (`pip install scikit-learn`).

---

## Como Executar

### Opção 1: Google Colab (recomendado)

1. Faça o download do dataset **DronePropB** no Mendeley Data e envie os arquivos `.mat` para uma pasta no seu Google Drive (por exemplo, `MyDrive/DronePropB`).
2. Abra o arquivo `projeto_simplificado.ipynb` no Google Colab.
3. Na **Célula 0 (Configuração)**, descomente as linhas de montagem do Google Drive e aponte `DATA_DIR` para a pasta dos `.mat`:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   DATA_DIR = Path("/content/drive/MyDrive/DronePropB")
   ```
4. Execute todas as células em ordem (**Ambiente de execução → Executar tudo**).

### Opção 2: Execução Local

1. Clone o repositório e acesse a pasta do projeto:
   ```bash
   git clone <url-do-repositorio>
   cd sinais-e-sistemas-2026.2
   ```
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt scikit-learn jupyter
   ```
4. Coloque os arquivos `.mat` do DronePropB na pasta `./dataset` na raiz do repositório (ou ajuste a variável `DATA_DIR` na Célula 0 do notebook).
5. Inicie o Jupyter e execute o notebook de cima para baixo:
   ```bash
   jupyter notebook projeto_simplificado.ipynb
   ```

---

## Conteúdo do Notebook (`projeto_simplificado.ipynb`)

| Seção | Conteúdo implementado |
| :--- | :--- |
| **0. Configuração** | Importações, caminho `DATA_DIR`, par fixo (`F0_SV0_SP1_C1_R1` $\times$ `F2_SV2_SP1_C1`), definição de janelas (4 s, 50%) e das 3 bandas (`B1`: 0–50 Hz, `B2`: 50–300 Hz, `B3`: 300–5000 Hz). |
| **3.1 Dados e janelamento** | Seleção dos 87 registros, verificação de duplicidade (`C2 == C1`), auditoria de qualidade (0 `NaN`/`Inf`, 0 lacunas), tabela de metadados e geração das 1.923 janelas preservando `record_id`. |
| **3.2 Domínio do tempo** | Gráfico temporal comparativo (4 s + zoom de 50 ms), extração das 7 características obrigatórias, tabela resumo, boxplots (`rms`, `std`, `pico_abs`, `crista`), verificação numérica de $\text{RMS}^2 = \mu^2 + \sigma^2$ e análise do peso da componente DC ($\mu^2/P$). |
| **3.3 Domínio da frequência (FFT)** | FFT unilateral (`rfft`), normalização de amplitude e potência com verificação de Parseval ($\sum P_{\text{bandas}} = \sigma^2$), espectros comparativos com picos anotados, 7 medidas espectrais, tabela resumo e boxplots. |
| **3.3.6 Resolução espectral** | Comparação de 4 s ($\Delta f = 0{,}25\text{ Hz}$) vs. 1 s ($\Delta f = 1{,}00\text{ Hz}$) na mesma janela com zoom de $\pm 15\text{ Hz}$ no pico dominante. |
| **3.3.7 Efeito de amostragem (*aliasing*)** | Decimação por fator 25 ($10\text{ kHz} \to 400\text{ Hz}$, Nyquist $= 200\text{ Hz}$) sem filtro (`x[::25]`) vs. com filtro anti-aliasing (`resample_poly`) e previsão analítica dos picos dobrados. |
| **3.4 Comparação entre domínios** | Tabela de AUC simétrica (Tempo $\times$ Frequência) por velocidade, matriz de correlação de Spearman (média intra-velocidade), identificação de pares redundantes ($|\rho| \ge 0{,}85$) e definição do conjunto de atributos para a Etapa 2. |
| **3.6 Investigação adicional** | Estudo do efeito da velocidade de rotação (`SP1` $\to$ `SP3`) sobre a frequência fundamental $f_0$ e verificação da estrutura harmônica ($f/f_0$). |

---

## Principais Achados da 1ª Entrega

* **Integridade dos dados:** os 87 registros selecionados não possuem valores inválidos nem lacunas temporais; a componente média (DC) responde por menos de $0{,}2\%$ da potência (em mediana), tornando $\text{RMS} \approx \sigma$.
* **Tempo vs. Frequência:** a velocidade de rotação domina a escala de amplitude (RMS cresce $\sim 7\times$ de `SP1` para `SP3`), exigindo comparação intra-velocidade. No tempo, o fator de crista é a única medida com sentido consistente nas três velocidades (defeituoso mais impulsivo). Na frequência, a entropia espectral (AUC média $0{,}77$) e a potência em `B3` ($0{,}70$) destacam-se pela consistência.
* **Estrutura harmônica:** a frequência fundamental saudável sobe de $55{,}75\text{ Hz}$ (`SP1`) para $98{,}50\text{ Hz}$ (`SP2`, $1{,}77\times$) e $133{,}50\text{ Hz}$ (`SP3`, $2{,}39\times$), arrastando seus múltiplos inteiros ao longo do espectro.
