## 📄 `README.md`

```markdown
# DronePropB - Análise de Vibração e Classificação de Hélices

Pipeline de processamento de sinais e extração de características de vibração de drones para classificação de hélices saudáveis e defeituosas.

## Requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd ML-Sinais
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # ou
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Certifique-se de que os dados do dataset `DronePropB Ground Testing Dataset` está na raiz do projeto em "dataset".

## Como executar

O pipeline é dividido em 4 etapas sequenciais. Execute na ordem:

### Etapa 3.1 — Preparação e Janelamento
Lê os arquivos `.mat`, remove duplicatas, verifica a qualidade do sinal e gera as janelas.
```bash
python preprocess3-1.py
```

### Etapa 3.2 — Extração de Características no Tempo
Calcula as 8 features de tempo (média, RMS, std, pico, crista, assimetria, curtose) e gera as tabelas de efeito.
```bash
python preprocess3-2.py
```

### Etapa 3.3 — Extração de Características na Frequência
Aplica FFT para extrair entropia, centroide, largura de banda, magnitude dominante e potência em bandas.
```bash
python preprocess3-3.py
```

### Etapa 3.4 — Comparação e Seleção de Features
Compara os domínios do tempo e frequência, analisa redundância e seleciona o conjunto final.
```bash
python preprocess3-4.py
```

## Resultados

Todas as saídas (CSVs, figuras e seleções) são geradas na pasta `outputs/`:

- `outputs/*.csv` — Tabelas de features, resumos e testes de efeito.
- `outputs/figuras/` — Gráficos da análise temporal.
- `outputs/figuras_espectrais/` — Gráficos da análise em frequência.
- `outputs/figuras_3_4/` — Gráficos comparativos e matriz de correlação.