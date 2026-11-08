# AGENTS.md — Guia Completo de Arquitetura e Operação do Projeto `BonsaiToClef`

Este documento serve como referência técnica e operacional definitiva para agentes de IA e desenvolvedores atuando no repositório **BonsaiToClef**. Todo trabalho de código, execução, avaliação ou extensão deve respeitar rigorosamente os contratos arquiteturais, científicos e metodológicos documentados aqui.

---

## 1. Visão Geral do Projeto

O repositório **BonsaiToClef** investiga, implementa, treina, calibra, quantiza (GGUF) e avalia **modelos de decisão baseados em LLMs causais que produzem distribuições de probabilidade em uma única passagem pelo backbone (`forward`), sem geração autorregressiva de tokens (`generate()`)**.

O projeto é estruturado em **quatro módulos/estágios evolutivos independentes**, cada um com seu próprio ambiente virtual (`.venv`) e `requirements.txt` estritamente fixados:

1. **Raiz (`/`) — Baseline Zero-Shot CLEF-Flash no BANKING77**:
   - Avalia o modelo oficial [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) de forma zero-shot na tarefa de classificação de 77 intenções bancárias ([`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77)).
   - Fornece o avaliador canônico offline (`avaliar.py`) sem dependência de GPU.
2. **`DecisionLFM/` — Protótipo Supervisionado Single-Task (`choice`)**:
   - Adapta o backbone [`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B) com uma cabeça de similaridade cosseno compartilhada (`DecisionHead`) e LoRA no modelo de linguagem.
   - Treina em duas fases no split `train` do BANKING77 (1 época apenas cabeça + 3 épocas cabeça + LoRA) com perda de Entropia Cruzada (CE) e compara diretamente contra o baseline CLEF.
3. **`DecisionLFM/DecisioLFM-Brier-Input/` — Sistema Multitarefa Tipado (`choice`, `noul`, `score`) com CE+Brier, Calibração e GGUF**:
   - Estende o `DecisionLFM` para entrada estruturada JSON (`state` + `questions` com múltiplas perguntas respondidas sequencialmente, uma passagem pelo backbone por pergunta).
   - Suporta três tipos de decisão: escolha categórica (`choice`), booleana/probabilidade de verdadeiro (`noul`) e escala ordinal com esperança matemática (`score`).
   - Treinamento conjunto multitarefa (até 3 épocas, LoRA rank 8) com função objetivo híbrida **CE + Brier multiclasses** (pesos diferenciados por tipo).
   - Particionamento estrito da validação em metade de **seleção** e metade de **calibração** (Temperature Scaling por tipo com opção de proteção do MAE de `score`).
   - Pipeline completo de fusão LoRA, exportação para **GGUF (`F16`, `Q8_0`, `Q4_0`)**, patch customizado no `llama.cpp` para omitir a projeção de vocabulário (`lm_head`), e bridge C++/GGML (`bridge.cpp` -> `libdecisionlfm.so`) para inferência híbrida CUDA + CPU.
4. **`DecisionLFM/DecisionLFM-QLoRA/` — Treinamento e Inferência com QLoRA (4-bit NF4 + Double Quantization)**:
   - Implementa o protocolo **QLoRA** (Dettmers et al., 2023) sobre o `LiquidAI/LFM3-VL-2B`: carrega o módulo de linguagem em 4-bit NormalFloat (`nf4`) com Dupla Quantização (`bnb_4bit_use_double_quant=True`), computação em `bfloat16` e exclusão explícita dos módulos de visão inativos (`llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]`).
   - Prepara o modelo quantizado com `prepare_model_for_kbit_training(..., use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})` mantendo a `DecisionHead` em FP32.
   - Suporta tanto o otimizador `AdamW` padrão do PyTorch (para comparação científica controlada contra o LoRA BF16) quanto os otimizadores paginados `PagedAdamW32bit` e `PagedAdamW8bit` do `bitsandbytes`.
   - Na exportação GGUF (`gguf/exportar.py`), funde os adaptadores QLoRA diretamente sobre o modelo-base original em **FP32 na CPU**, evitando ruído de dupla quantização (`NF4 -> GGUF`), e inclui o comparador pareado `comparar_qlora.py` (LoRA BF16 vs QLoRA NF4).

---

## 2. Mapa de Diretórios e Módulos

```text
BonsaiToClef/
├── README.md                          # Documentação e resultados do baseline CLEF-Flash no BANKING77
├── requirements.txt                   # Dependências do baseline CLEF (transformers==5.10.2, torch==2.11.0)
├── inferir.py                         # Inferência zero-shot do Cloudflare/clef-flash (batch 1, BF16)
├── avaliar.py                         # Avaliador canônico CPU-only (gera metrics.json a partir de predictions.jsonl)
│
└── DecisionLFM/                       # Estágio 1: DecisionLFM para escolha única (BANKING77)
    ├── README.md                      # Guia de treino, verificação, inferência e comparação no BANKING77
    ├── AVALIACAO.md                   # Estudo arquitetural prévio do backbone LiquidAI/LFM3-VL-2B e licença
    ├── requirements.txt               # Dependências DecisionLFM (transformers==5.21.0, peft==0.21.2, torch==2.11.0)
    ├── dados.py                       # Loader fixado do PolyAI/banking77 com descrições idênticas ao CLEF
    ├── modelo.py                      # Codificador por offsets, collate, DecisionHead, DecisionLFM, LoRA e I/O
    ├── treinar.py                     # Treino em 2 fases (head -> head+LoRA) com split estratificado 90/10
    ├── verificar.py                   # Replay determinístico da validação e checagem de 1 forward call por decisão
    ├── inferir.py                     # Inferência compatível com o protocolo e avaliador da raiz
    ├── comparar.py                    # Comparação pareada estrita entre execuções CLEF e DecisionLFM
    ├── test_modelo.py                 # Testes unitários de tokenização, offsets unicode, máscaras e invariância de ordem
    ├── test_comparar.py               # Testes unitários do validador de comparação
    │
    └── DecisioLFM-Brier-Input/        # Estágio 2: Multitarefa (choice/noul/score), JSON, CE+Brier, Calibração e GGUF
        ├── README.md                  # Documentação completa do fluxo PyTorch, Calibração, Busca e GGUF
        ├── requirements.txt           # Mesma base do DecisionLFM (transformers==5.21.0, peft==0.21.2, torch==2.11.0)
        ├── esquema.py                 # Validação do contrato JSON de entrada e formatação de saídas (choice/noul/score)
        ├── modelo.py                  # DecisionLFM com suporte a CUDA/MPS, temperaturas por tipo e método responder()
        ├── objetivo.py                # Função de perda decision_loss (Cross-Entropy + Brier na escala [0, 2])
        ├── dados_tipos.py             # Preparador do benchmark de 9 datasets (6 in-domain, 3 held-out) com deduplicação por grupo
        ├── manifesto.py               # Criador e validador de manifest.json para datasets JSONL customizados
        ├── treinar_tipos.py           # Treino conjunto (até 3 épocas), batches agrupados por comprimento, seleção/early-stop
        ├── verificar_checkpoint.py    # Replay exato da época selecionada na validação após recarregar pesos do disco
        ├── calibracao.py              # Partição determinística da validação, busca em grade de Temperatura por tipo e seleção
        ├── calibrar.py                # Extrai logits da validação e gera calibration.json (com opção --protect-score-mae)
        ├── ajustar_brier.py           # Pipeline automatizado de busca de hiperparâmetros focada em Brier
        ├── resumir_calibracao.py      # Auditoria pareada antes/depois da calibração (gera COMPARACAO_CALIBRACAO.md)
        ├── comparar_ajustes.py        # Busca ordenada mudando 1 hiperparâmetro por vez, priorizando acurácia e protegendo MAE
        ├── resumir_ajustes.py         # Auditoria completa da busca ordenada (gera COMPARACAO_AJUSTES.md e hashes finais)
        ├── decidir_json.py            # CLI de inferência PyTorch para arquivos JSON com múltiplas perguntas
        ├── avaliar_tipos.py           # Benchmark PyTorch (DecisionLFM ou CLEF-Flash) no conjunto de teste multitarefa
        ├── metricas.py                # Cálculo padronizado de acurácia, Macro-F1 por tarefa, Brier, MAE de score, latência e VRAM
        ├── comparar.py                # Comparador pareado CLEF vs DecisionLFM com análise por faixas de tokens (buckets)
        ├── exemplos/
        │   ├── entrada.json           # Exemplo de payload JSON contendo perguntas choice, noul e score
        │   └── fontes_benchmark.json  # SHAs fixados dos 9 datasets do Hugging Face para reprodução exata
        ├── gguf/                      # Subsistema de exportação e inferência nativa C++/CUDA/GGML
        │   ├── requirements.txt       # Dependência para monitoramento NVML (nvidia-ml-py==13.590.48)
        │   ├── preparar_backend.sh    # Clona llama.cpp em commit fixo, aplica patch LFM3 e compila libdecisionlfm.so
        │   ├── llama-embedding-only.patch # Patch em src/models/lfm3.cpp para pular a projeção de vocabulário quando embeddings=true
        │   ├── bridge.cpp             # ABI C que executa o backbone GGUF na GPU e a DecisionHead em GGML/CPU (ou CLEF nativo)
        │   ├── runtime.py             # Wrapper ctypes para libdecisionlfm.so + amostrador de VRAM via NVML a cada 2 ms
        │   ├── exportar.py            # Fusão LoRA em FP32 na CPU, exportação de backbone e cabeça para F16, Q8_0 e Q4_0
        │   ├── decidir_json.py        # CLI de inferência JSON usando os arquivos GGUF
        │   └── avaliar.py             # Avaliador de benchmark para modelos GGUF (DecisionLFM e CLEF)
        └── tests/                     # Suíte de 32 testes unitários rápidos (não exigem carregar os pesos do backbone)
            ├── test_modelo.py
            ├── test_json.py
            ├── test_objetivo.py
            ├── test_calibracao.py
            ├── test_multitarefa.py
            ├── test_manifesto.py
            └── test_utilitarios.py
    │
    └── DecisionLFM-QLoRA/             # Estágio 3: QLoRA (4-bit NF4 + Double Quantization + Paged Optimizers)
        ├── README.md                  # Guia completo de treino QLoRA, comparação LoRA vs QLoRA e GGUF
        ├── requirements.txt           # Adiciona bitsandbytes==0.50.2 às dependências do DecisionLFM
        ├── modelo.py                  # BitsAndBytesConfig (NF4 + DQ), prepare_model_for_kbit_training e I/O 4-bit
        ├── treinar_tipos.py           # Treino QLoRA com suporte a adamw, paged_adamw_32bit e paged_adamw_8bit
        ├── comparar_qlora.py          # Comparador pareado LoRA BF16 vs QLoRA 4-bit NF4 (teste + VRAM de treino)
        ├── gguf/exportar.py           # Fusão limpa dos adaptadores QLoRA sobre o backbone FP32 na CPU
        └── tests/test_qlora.py        # Testes unitários específicos do QLoRA (totalizando 37 testes na pasta)
```

---

## 3. Arquitetura Técnica em Detalhes

### 3.1. Backbone e Alvos LoRA (`LiquidAI/LFM3-VL-2B`)
- **Revisão Fixada**: `f83d2b0a2296aaa277f394990a9b8b949ddf9691`.
- **Estrutura**: Modelo multimodal de ~1,92B parâmetros totais, sendo ~1,73B no módulo de linguagem (`language_model`), que possui 24 camadas híbridas (13 blocos de convolução curta, 5 atenções locais e 6 atenções globais) e `hidden_size = 2048`.
- **Execução Sem Geração**: Chamamos `backbone(input_ids=..., attention_mask=..., use_cache=False, return_dict=True).last_hidden_state`. Não usamos `lm_head`, projeção no vocabulário, KV cache ou `generate()`.
- **Configuração LoRA**:
  - Aplicada exclusivamente às camadas lineares do modelo de linguagem via regex:
    `r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"`
  - Padrão: `r=8`, `lora_alpha=16` (`2 * rank`), `lora_dropout=0.05`.
  - Ativa `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` durante o treino LoRA.

### 3.2. Codificação do Prompt e Alinhamento de Spans (`encode` e `collate`)
Como o backbone LFM3 é **causal**, os tokens de uma alternativa não podem ver o estado final se vierem depois dele, e o token de decisão precisa ver todas as alternativas, a pergunta e a situação (`state`). Por isso, a estrutura do prompt é construída nesta ordem exata:

```text
{question}
Options:
- {descrição_opção_1}
- {descrição_opção_2}
...

State:
{state}
Choose exactly one option.
```

1. **IDs Ocultos**: As chaves/IDs das alternativas (`option_ids`) **nunca entram no texto do prompt**; apenas suas descrições textuais entram após `"- "`. Isso evita viés de rótulo lexical e garante que o modelo avalie a semântica das descrições.
2. **Mapeamento de Posições por Offsets**:
   - O conteúdo é formatado com `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`.
   - A tokenização usa `return_offsets_mapping=True` (exige `tokenizer.is_fast`).
   - Para cada alternativa $j$, identificamos os índices exatos dos tokens cujos intervalos de caracteres se sobrepõem ao span da descrição $j$. Se alguma alternativa ficar sem tokens, um `ValueError` é lançado.
3. **Sem Truncamento Silencioso**: Se `len(input_ids) > max_length`, `encode()` lança `ValueError` imediatamente. Entradas longas jamais são truncadas silenciosamente.
4. **Pooling e Collate (`collate`)**:
   - Faz padding à direita nos `input_ids` até o maior comprimento do batch.
   - Constrói `option_weights` de dimensão `(batch, max_choices, seq_len)`, onde cada token pertencente à alternativa $j$ recebe peso uniforme $1 / |P_j|$.
   - Constrói `option_mask` booleana `(batch, max_choices)` indicando quais alternativas são válidas para cada exemplo do batch (permitindo misturar exemplos com números diferentes de alternativas no mesmo batch, ex.: 77 classes de `choice`, 2 de `noul` e 5 de `score`).

### 3.3. Cabeça de Decisão (`DecisionHead`)
A cabeça opera sempre em **FP32** (mesmo com o backbone em BF16) e possui `head_size = 256` (1.052.673 parâmetros para `hidden_size = 2048`):

1. **Vetor de Consulta (`query`)**: Extrai o estado oculto do **último token válido** da sequência (`query_index = attention_mask.sum(-1) - 1`).
2. **Vetores das Alternativas (`options`)**: Calcula a média dos estados ocultos dos tokens de cada alternativa via multiplicação de matrizes em batch: `torch.bmm(option_weights, hidden)`.
3. **Pontuação por Similaridade Cosseno Escalada**:
   $$\tilde{q} = \text{Normalize}(\text{Linear}_{\text{query}}(\text{LayerNorm}(q))), \quad \tilde{k}_c = \text{Normalize}(\text{Linear}_{\text{option}}(\text{LayerNorm}(k_c)))$$
   $$\text{logits}_{b, c} = (\tilde{q}_b \cdot \tilde{k}_{b, c}) \times \min(\exp(\text{log\_scale}), 100.0)$$
   onde `log_scale` é um parâmetro treinável inicializado em $\ln(10.0)$.
4. **Mascaramento**: Posições onde `~option_mask` recebem `-inf` antes do `softmax`, garantindo probabilidade zero e gradiente zero para alternativas de padding.
5. **Equivariância à Permutação na Cabeça**: Para vetores de entrada fixos, permutar a ordem das alternativas permuta identicamente os logits gerados pela cabeça.

### 3.4. Tipos de Decisão e Contrato JSON (`DecisioLFM-Brier-Input`)
O módulo `esquema.py` valida entradas no formato `{"state": ..., "questions": {id: spec}}`:
- **`state`**: Se for string, é mantido; se for objeto/lista/primitivo JSON, é serializado deterministicamente com `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
- **`choice`**: Requer `criteria` como dicionário `{id: descrição}` com $\ge 2$ opções. As chaves são ordenadas alfabeticamente (`dict(sorted(criteria.items()))`) na preparação para garantir determinismo na inferência, mas são embaralhadas durante o treino. Retorna `{"type": "choice", "choice": argmax, "confidence": max_p, "probabilities": {...}}`.
- **`noul`** (Booleano / Probabilidade de Verdadeiro): Define descrições padrão para `"true"` (`"The proposition is true or the answer is yes."`) e `"false"` (`"The proposition is false or the answer is no."`), permitindo override parcial via `criteria`. Retorna `{"type": "noul", "noul": P("true")}`.
- **`score`** (Ordinal): Requer `criteria` como lista ordenada de $\ge 2$ strings representando níveis de `0` a `N-1`. Os IDs internos passam a ser `"0"`, `"1"`, ..., `"N-1"`. Retorna o valor esperado contínuo:
  $$\text{score} = \sum_{i=0}^{N-1} i \cdot P(i)$$
  junto com `confidence`, `probabilities` e `legend`.

### 3.5. Função Objetivo Híbrida (`objetivo.py`)
Em `DecisioLFM-Brier-Input`, a loss combina Entropia Cruzada (CE) e Brier Score multiclasses sobre as alternativas válidas:
$$\mathcal{L}(z, y) = \text{CE}(z, y) + w_{\text{brier}} \sum_{c=1}^{C} (p_c - \mathbb{I}[c = y])^2, \quad p = \text{softmax}(z)$$
- **Escala do Brier**: Soma direta sobre as classes sem dividir por $C$ e sem fator $1/2$, variando no intervalo $[0, 2]$.
- **Pesos por Tipo**: Por padrão, `brier_weight = 1.0` para `choice` e `noul`, e `score_brier_weight = 2.0` para `score`. Configurar ambos como `0` recupera exatamente a entropia cruzada padrão.
- **Acumulação de Gradientes**: Tanto em `treinar.py` quanto em `treinar_tipos.py`, a loss por microbatch é calculada com `reduction="sum"`, acumulada e dividida pelo número exato de exemplos acumulados (`accumulated_samples`) antes do `clip_grad_norm_(..., 1.0)` e `optimizer.step()`. Isso preserva o gradiente exato do batch completo mesmo quando o último microbatch tem tamanho menor.

### 3.6. Particionamento de Dados, Validação e Calibração (`calibracao.py`)
Para evitar qualquer vazamento de informação (*data leakage*) ou sobreajuste na validação:
1. **Agrupamento por Documento (`group_key`)**: Textos normalizados (ou a premissa no SNLI/RTE e a passagem no BoolQ) geram um hash SHA-256 (`group`). Nenhum `group` pode aparecer em mais de um split (`train`, `validation`, `test`).
2. **Tarefas Reservadas (`held_out_task`)**: No benchmark de 9 fontes (`dados_tipos.py`):
   - **In-domain (6 tarefas, presentes em train/val/test)**: `banking77`, `ag_news` (`choice`); `boolq`, `snli` (`noul`); `sst5`, `yelp` (`score`).
   - **Held-out (3 tarefas, presentes APENAS em `test`)**: `emotion` (`choice`), `rte` (`noul`), `amazon` (`score`).
3. **Divisão da Validação (50% Seleção / 50% Calibração)**:
   - `validation_partition()` divide os grupos de documentos de cada tarefa na validação em duas metades disjuntas: `selection` e `calibration`.
   - **`selection`**: Usada exclusivamente durante o treino para escolher a melhor época, acionar `ReduceLROnPlateau` / early stopping e escolher hiperparâmetros entre modelos candidatos.
   - **`calibration`**: Usada exclusivamente após o treino (`calibrar.py`) para ajustar uma temperatura escalar $T_{\text{tipo}} \in [0.25, 16.0]$ para cada tipo (`choice`, `noul`, `score`) minimizando o Brier sobre os logits congelados.
4. **Proteção do MAE em `score` (`--protect-score-mae`)**:
   - Mudar a temperatura não altera o `argmax` (logo não muda a acurácia), mas altera a distribuição de probabilidades e, portanto, desloca a esperança matemática $\sum i \cdot P(i)$ nas tarefas `score`.
   - Com `--protect-score-mae`, uma temperatura $T \neq 1$ só é aceita para `score` se reduzir o Brier **e** não aumentar o MAE na metade de calibração em relação a $T = 1$.
5. **Verificação Criptográfica de Pesos**: `calibration.json` armazena o `weight_hash` (SHA-256 de `head.safetensors`, adaptadores LoRA e configs). Ao carregar um modelo com `DecisionLFM.load(..., calibrated=True)`, se os pesos tiverem mudado em relação ao `calibration.json`, um `ValueError` é lançado.

### 3.7. Pipeline GGUF e Bridge C++/CUDA (`gguf/`)
Para execução de baixa latência e baixo consumo de VRAM fora do PyTorch:
1. **Patch no `llama.cpp` (`llama-embedding-only.patch`)**:
   - Aplicado sobre o commit fixo `0ec4027a2a0183de87c007069adddc06e665f7de` em `src/models/lfm3.cpp`.
   - Condiciona a multiplicação final pela matriz de vocabulário (`model.output`) a `if (!cparams.embeddings)`, economizando computação e memória já que o `DecisionLFM` consome apenas `result_norm` (`last_hidden_state`).
2. **Exportação (`gguf/exportar.py`)**:
   - Carrega o backbone base em FP32 na CPU, funde o adaptador LoRA com `merge_and_unload(safe_merge=True)`, extrai `model.language_model` e converte para `backbone-f16.gguf` via `convert_hf_to_gguf.py`.
   - Quantiza o backbone para `Q8_0` e `Q4_0` (usando `--pure --output-tensor-type q4_0 --token-embedding-type q4_0` no Q4).
   - Exporta a cabeça (`head.safetensors`) diretamente para `head-{f16,q8_0,q4_0}.gguf`, quantizando `query.weight` e `option.weight` enquanto mantém `norm.weight`, `norm.bias` e `log_scale` em FP32.
3. **Execução na Bridge (`gguf/bridge.cpp`)**:
   - Inicializa o `llama_context` com `n_gpu_layers = -1` (todas as camadas na GPU CUDA), `embeddings = true`, `pooling_type = LLAMA_POOLING_TYPE_NONE`, `attention_type = LLAMA_ATTENTION_TYPE_CAUSAL` e Flash Attention habilitado.
   - No modo **DecisionLFM**, extrai os embeddings de todos os tokens do prompt da GPU, faz o pooling médio dos spans das alternativas e a LayerNorm em C++, e executa as projeções matriciais da cabeça (`ggml_mul_mat`) no backend GGML CPU com 8 threads, retornando os scores escalados por cosseno.
   - No modo **CLEF**, configura `llama_batch_ext_set_decision_order` para cada span e lê os scores diretamente do grafo CUDA nativo do CLEF no `llama.cpp`.

---

## 4. Ambientes, Dependências e Hardware

### Isolamento de Ambientes Virtuais
**REGRA CRÍTICA**: Nunca misture os ambientes virtuais da raiz e das subpastas `DecisionLFM`.
- **Raiz (`/requirements.txt`)**: Usa `transformers==5.10.2` e `tokenizers==0.22.2` (exigidos pelo código oficial do `Cloudflare/clef-flash`).
- **`DecisionLFM/` e `DecisioLFM-Brier-Input/`**: Usam `transformers==5.21.0`, `peft==0.21.2` e `tokenizers==0.23.3` (exigidos pelo `LiquidAI/LFM3-VL-2B`).
- **`DecisioLFM-Brier-Input/gguf/requirements.txt`**: Adiciona `nvidia-ml-py==13.590.48` ao ambiente de `DecisioLFM-Brier-Input` para medição de VRAM do processo via NVML.

### Requisitos de Hardware
- **Raiz e `DecisionLFM/`**: Exigem GPU NVIDIA com CUDA e suporte a `bfloat16` (validado em RTX 4090 de 24 GB). `avaliar.py` na raiz roda apenas em CPU.
- **`DecisioLFM-Brier-Input/` (PyTorch)**: Suporta GPU CUDA com `bfloat16` ou Apple Silicon (`mps`).
- **`DecisioLFM-Brier-Input/gguf/`**: Exige Linux, GPU CUDA, CMake, compilador C++17 e CUDA Toolkit (`nvcc`).

---

## 5. Comandos Principais de Operação e Testes

### 5.1. Executar Testes Automatizados (Rápidos / Sem GPU Pesada)
Os testes unitários carregam apenas o tokenizer (e cabeças sintéticas pequenas), sem baixar ou instanciar os ~3,8 GB de pesos do backbone:

```bash
# Testes do estágio 1 (DecisionLFM)
cd DecisionLFM
.venv/bin/python -m unittest -v test_modelo test_comparar

# Testes do estágio 2 (DecisioLFM-Brier-Input — 32 testes)
cd DecisionLFM/DecisioLFM-Brier-Input
.venv/bin/python -m unittest discover -s tests -v

# Usando tokenizer já salvo localmente e modo 100% offline:
HF_HUB_OFFLINE=1 DECISIONLFM_TEST_TOKENIZER=/caminho/para/checkpoint/tokenizer \
  .venv/bin/python -m unittest discover -s tests -v
```

### 5.2. Fluxo 1: Baseline Zero-Shot CLEF-Flash (Raiz)
```bash
# 1. Teste rápido de desenvolvimento (20 amostras do train)
python inferir.py --split train --limit 20 --seed 42 --output-dir resultados/train-20
python avaliar.py resultados/train-20

# 2. Avaliação oficial completa (3.080 amostras do test)
python inferir.py --split test --output-dir resultados/test
python avaliar.py resultados/test
```

### 5.3. Fluxo 2: DecisionLFM no BANKING77 (`DecisionLFM/`)
```bash
cd DecisionLFM

# 1. Treino supervisionado (1 época cabeça + 3 épocas cabeça+LoRA, usando apenas train)
.venv/bin/python -u treinar.py --output-dir modelos/banking77-novo

# 2. Verificar integridade do checkpoint (replay da validação + teste de 1 forward call)
.venv/bin/python verificar.py --checkpoint modelos/banking77-novo

# 3. Inferência no split test e avaliação com o avaliador da raiz
.venv/bin/python inferir.py --checkpoint modelos/banking77-novo --split test --output-dir resultados/test-novo
.venv/bin/python ../avaliar.py resultados/test-novo

# 4. Comparar pareado contra o baseline CLEF-Flash
.venv/bin/python comparar.py ../resultados/test resultados/test-novo
```

### 5.4. Fluxo 3: Multitarefa Tipado, Calibração e GGUF (`DecisionLFM/DecisioLFM-Brier-Input/`)
```bash
cd DecisionLFM/DecisioLFM-Brier-Input

# 1. Preparar dados das 9 fontes com revisões fixadas
.venv/bin/python dados_tipos.py --output-dir dados/multitarefa --sources exemplos/fontes_benchmark.json --seed 42

# 2. Treinar (até 3 épocas conjuntas, CE+Brier, LoRA rank 8)
.venv/bin/python treinar_tipos.py --data-dir dados/multitarefa --output-dir modelos/r8-novo \
  --epochs 3 --lora-rank 8 --brier-weight 1 --score-brier-weight 2 \
  --head-lr 0.0003 --lora-lr 0.00003 --batch-size 4 --gradient-accumulation 4

# 3. Replay de verificação da época selecionada na validação
.venv/bin/python verificar_checkpoint.py --checkpoint modelos/r8-novo \
  --data-dir dados/multitarefa --output resultados/replay-r8-novo.json

# 4. Calibrar temperaturas por tipo na metade reservada da validação (protegendo MAE de score)
.venv/bin/python calibrar.py --checkpoint modelos/r8-novo --data-dir dados/multitarefa --protect-score-mae

# 5. Inferência direta em arquivo JSON com múltiplas perguntas
.venv/bin/python decidir_json.py --checkpoint modelos/r8-novo --input exemplos/entrada.json

# 6. Avaliar no teste em PyTorch
.venv/bin/python avaliar_tipos.py --checkpoint modelos/r8-novo --data-dir dados/multitarefa --output-dir resultados/teste-r8-novo

# 7. Compilar backend GGUF, exportar F16/Q8_0/Q4_0 e avaliar em GGUF
bash gguf/preparar_backend.sh
.venv/bin/python gguf/exportar.py --checkpoint modelos/r8-novo --output-dir modelos/gguf-decision --quants q8_0 q4_0
.venv/bin/python gguf/avaliar.py --family decision --quantization Q4_0 \
  --backbone modelos/gguf-decision/backbone-q4_0.gguf \
  --head modelos/gguf-decision/head-q4_0.gguf --tokenizer modelos/r8-novo/tokenizer \
  --data-dir dados/multitarefa --output resultados/decision-q4
```

### 5.5. Fluxo 4: QLoRA 4-bit NF4 e Comparação Pareada (`DecisionLFM/DecisionLFM-QLoRA/`)
```bash
cd DecisionLFM/DecisionLFM-QLoRA

# 1. Executar os 37 testes unitários (incluindo suíte test_qlora.py)
.venv/bin/python -m unittest discover -s tests -v

# 2. Treinar com QLoRA 4-bit NF4 + Double Quantization (reutilizando dados/multitarefa)
.venv/bin/python treinar_tipos.py \
  --data-dir ../DecisioLFM-Brier-Input/dados/multitarefa \
  --output-dir modelos/qlora-r8 \
  --epochs 3 --lora-rank 8 --brier-weight 1 --score-brier-weight 2 \
  --head-lr 0.0003 --lora-lr 0.00003 --batch-size 4 --gradient-accumulation 4 \
  --optimizer adamw

# (Opcional) Treino com PagedAdamW32bit e max-grad-norm=0.3 do artigo QLoRA:
.venv/bin/python treinar_tipos.py \
  --data-dir ../DecisioLFM-Brier-Input/dados/multitarefa \
  --output-dir modelos/qlora-paged-r8 \
  --epochs 3 --lora-rank 8 --brier-weight 1 --score-brier-weight 2 \
  --head-lr 0.0003 --lora-lr 0.00003 --batch-size 4 --gradient-accumulation 4 \
  --optimizer paged_adamw_32bit --max-grad-norm 0.3

# 3. Verificar replay da validação, calibrar e avaliar no teste
.venv/bin/python verificar_checkpoint.py --checkpoint modelos/qlora-r8 \
  --data-dir ../DecisioLFM-Brier-Input/dados/multitarefa --output resultados/replay-qlora-r8.json
.venv/bin/python calibrar.py --checkpoint modelos/qlora-r8 \
  --data-dir ../DecisioLFM-Brier-Input/dados/multitarefa --protect-score-mae
.venv/bin/python avaliar_tipos.py --checkpoint modelos/qlora-r8 \
  --data-dir ../DecisioLFM-Brier-Input/dados/multitarefa --output-dir resultados/teste-qlora-r8

# 4. Comparar pareado LoRA (BF16) vs QLoRA (4-bit NF4)
.venv/bin/python comparar_qlora.py \
  --lora ../DecisioLFM-Brier-Input/resultados/teste-r8-novo \
  --qlora resultados/teste-qlora-r8 \
  --output-dir resultados/comparacao-lora-vs-qlora
```

---

## 6. Regras de Ouro e Invariantes do Projeto (Diretrizes para Agentes)

Qualquer modificação ou novo script neste repositório **DEVE** preservar as seguintes invariantes de engenharia e rigor científico:

1. **Proibição de Sobrescrita de Diretórios**:
   - Todos os scripts de preparação de dados, treinamento, exportação, calibração, inferência, avaliação e comparação verificam `if output_dir.exists(): parser.error(...)`.
   - **Nunca** remova essa proteção nem sobrescreva resultados ou checkpoints existentes silenciosamente. Sempre utilize novos nomes de diretórios para novas execuções.

2. **Proibição de Truncamento Silencioso**:
   - Entradas que excedam `max_length` (8.192 na raiz/`DecisionLFM`, 1.024 no preparo multitarefa, 2.048 na inferência GGUF) **devem lançar `ValueError`** (ou ser explicitamente contabilizadas como `overlength` na criação do dataset). Nunca trunque o prompt ou o `state` silenciosamente, pois o estado da decisão ou as alternativas seriam corrompidos.

3. **Isolamento Estrito do Conjunto de Teste (`test`)**:
   - O split `test` jamais pode ser lido durante o treinamento, seleção de época, early stopping, escolha de hiperparâmetros ou ajuste de temperatura de calibração.
   - Em `DecisioLFM-Brier-Input`, mantenha sempre a separação entre as duas metades da validação (`selection` para escolha de época/modelo e `calibration` para ajuste de temperatura $T$).

4. **Verificação de Integridade via SHA-256**:
   - Os manifestos (`manifest.json`, `training.json`, `calibration.json`, `run.json`, `experiment.json`) gravam hashes SHA-256 dos scripts Python (`code_hashes`), dos datasets JSONL e dos pesos (`checkpoint_hash` e `weight_hash`).
   - Ao coletar hashes de código (`code_hashes` em `dados_tipos.py`), percorra apenas os diretórios do código-fonte (`.`, `gguf/`, `tests/`), **nunca** percorrendo `.venv/`, `modelos/`, `resultados/` ou `backends/`.
   - Ao calcular o `checkpoint_hash`, ignore sempre a subpasta `checkpoints/` (checkpoints intermediários de época/batch).

5. **Definições Exatas das Métricas**:
   - **Acurácia e Macro-F1**: Incluem falhas de execução (`error != None`) como previsões incorretas (`"__error__"`). No benchmark multitarefa (`metricas.py`), o Macro-F1 global é a **média não ponderada** dos Macro-F1 de cada tarefa individual.
   - **Brier Multiclasses**: Calculado apenas sobre registros válidos, como a **soma** dos erros quadráticos sobre todas as alternativas válidas: $\sum_c (p_c - y_c)^2 \in [0, 2]$ (sem dividir pelo número de classes e sem multiplicar por $1/2$).
   - **Score MAE**: Nas tarefas do tipo `score`, a acurácia e o F1 usam o `argmax` das probabilidades, enquanto o MAE avalia o erro absoluto entre a esperança matemática $\sum_{i=0}^{N-1} i \cdot P(i)$ e o índice inteiro alvo.
   - **Contadores de Memória Distintos**:
     - PyTorch CUDA: `torch.cuda.max_memory_allocated()` (pico de tensores alocados, em GiB).
     - GGUF CUDA: Máximo observado da memória do processo via NVML (`pynvml`, amostrado a cada 2 ms, inclui contexto CUDA) e breakdown de buffers GGML (`df_vram`).
     - Apple MPS: `torch.mps.current_allocated_memory()` após sincronização (memória unificada, reportada em `max_observed_mps_allocated_gib`, nunca rotulada como VRAM dedicada).
     - `comparar.py` recusa comparar execuções com contadores de memória incompatíveis.

6. **Determinismo e Precisão Numérica**:
   - Nunca serialize `NaN` ou `Infinity` em arquivos JSON/JSONL: utilize sempre `allow_nan=False` no `json.dumps()`.
   - Mantenha a cabeça `DecisionHead` e o cálculo da loss (`decision_loss`) em `float32` (e o ajuste de temperatura em `float64`), mesmo quando o backbone roda em `bfloat16`.
   - Preserve os commits e revisões fixados:
     - `Cloudflare/clef-flash`: `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`
     - `LiquidAI/LFM3-VL-2B`: `f83d2b0a2296aaa277f394990a9b8b949ddf9691`
     - `PolyAI/banking77`: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
     - `ggml-org/llama.cpp`: `0ec4027a2a0183de87c007069adddc06e665f7de`

7. **Licenciamento**:
   - Os pesos do `LiquidAI/LFM3-VL-2B` estão sob a **LFM Open License v1.0** (não é Apache 2.0; possui restrições comerciais acima de US$ 10M de receita anual e exige preservação dos avisos de licença ao redistribuir derivados).

