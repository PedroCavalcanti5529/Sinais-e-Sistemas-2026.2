## 📄 `README.md`

# DronePropB - Análise de Vibração e Classificação de Hélices

Pipeline de processamento de sinais e extração de características de vibração de drones para classificação de hélices saudáveis e defeituosas.

## Requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/PedroCavalcanti5529/Sinais-e-Sistemas-2026.2.git
   cd Sinais-e-Sistemas-2026.2
   ```

2. Crie e ative um ambiente virtual:
   * Linux/macOS
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   * Windows
   ```bash
   .venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Faça o download dos dados do dataset [DronePropB Ground Testing Dataset](https://data.mendeley.com/datasets/xkvfjmm8zg/1), extraia tudo do .zip, crie a pasta `./dataset` na raiz do projeto, e coloque todos os arquivos dentro dela.

## Como executar

O pipeline é dividido em 4 etapas sequenciais. Execute na ordem:

### Etapa 3.1 — Preparação e Janelamento
Lê os arquivos `.mat`, remove duplicatas, verifica a qualidade do sinal e gera as janelas.
```bash
python3 preprocess3_1.py
```

### Etapa 3.2 — Extração de Características no Tempo
Calcula as 8 features de tempo (média, RMS, std, pico, crista, assimetria, curtose) e gera as tabelas de efeito.
```bash
python3 preprocess3_2.py
```

### Etapa 3.3 — Extração de Características na Frequência
Aplica FFT para extrair entropia, centroide, largura de banda, magnitude dominante e potência em bandas.
```bash
python3 preprocess3_3.py
```

### Etapa 3.4 — Comparação e Seleção de Features
Compara os domínios do tempo e frequência, analisa redundância e seleciona o conjunto final.
```bash
python3 preprocess3_4.py
```

## Resultados

Todas as saídas (CSVs, figuras e seleções) são geradas na pasta `outputs/`:

- `outputs/*.csv` — Tabelas de features, resumos e testes de efeito.
- `outputs/figuras/` — Gráficos da análise temporal.
- `outputs/figuras_espectrais/` — Gráficos da análise em frequência.
- `outputs/figuras_3_4/` — Gráficos comparativos e matriz de correlação.
