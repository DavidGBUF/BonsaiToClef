# DecisionLFM — CE+Brier e entrada JSON

Esta pasta contém a abordagem `DecisioLFM-Brier-Input`, separada da implementação original em `DecisionLFM/`. Ela usa o backbone **LiquidAI/LFM3-VL-2B**, a cabeça cosseno compartilhada de alternativas e LoRA. A saída é uma decisão tipada, sem gerar texto.

O modelo completo tem aproximadamente 1,92 bilhão de parâmetros; o componente textual tem 1,73 bilhão. Nos experimentos GGUF é usado o componente textual. O modelo-base e o tokenizer estão fixados na revisão `f83d2b0a2296aaa277f394990a9b8b949ddf9691`.

## Diferenças em relação ao código commitado

A referência é o commit [`ac1b21a`](https://github.com/VeigarGit/BonsaiToClef/commit/ac1b21a73e86cd734082107acdf505be9f748955), que adicionou o DecisionLFM original ao BonsaiToClef.

| Aspecto | DecisionLFM original | Esta abordagem |
|---|---|---|
| Entrada | Texto, instrução e alternativas | JSON com `state` e `questions`, identificadas por IDs |
| Resposta | Escolha entre alternativas | `choice`, `noul` e `score` |
| Várias perguntas | Uma por chamada | Várias no JSON; uma passagem pelo backbone por pergunta |
| Dados de treino | BANKING77 | Várias tarefas dos três tipos, com classes e critérios variáveis |
| Treinamento padrão | Uma época da cabeça e três de LoRA | Três épocas conjuntas de cabeça + LoRA, rank 8 |
| Objetivo | Entropia cruzada — CE | CE + 1,0 × Brier multiclasses |
| Execução | PyTorch BF16, cabeça FP32 | PyTorch e GGUF F16/Q8_0/Q4_0 |
| Avaliação | BANKING77 | Coorte pareada, métricas por tarefa/tipo e faixas de tokens |

A arquitetura do backbone e do scorer é a mesma. A loss acrescenta o Brier às probabilidades das alternativas válidas:

```text
p = softmax(logits)
loss = CE(logits, alvo) + peso_brier × soma((p - one_hot(alvo))²)
```

O peso padrão é `1.0`; `--brier-weight 0` recupera CE. O Brier usa a escala `[0, 2]`, sem fator `1/2`. Os três tipos são treinados como distribuições sobre alternativas; `score` preserva a ordem dos níveis e retorna a esperança do índice.

### Relação com o CLEF-Flash

O BonsaiToClef também mantém o pipeline do checkpoint publicado `Cloudflare/clef-flash`. Nesta abordagem, o DecisionLFM usa LFM3 e a cabeça cosseno; o CLEF mantém seu backbone e sua cabeça conjunta originais. O contrato JSON é semelhante, mas o DecisionLFM faz uma passagem por pergunta, enquanto a cabeça do CLEF processa os campos conjuntamente.

No benchmark, o CLEF é usado com os pesos publicados; o DecisionLFM recebe fine-tuning nos 12.000 exemplos preparados. Os resultados comparam esses pipelines e regimes de treinamento. Eles não isolam o efeito da arquitetura ou da loss individualmente.

## Organização

```text
DecisioLFM-Brier-Input/
├── esquema.py                 # validação JSON e respostas tipadas
├── modelo.py                  # backbone, LoRA, pooling e scorer original
├── objetivo.py                # CE + Brier
├── dados_tipos.py             # preparação do benchmark multitarefa
├── manifesto.py               # manifesto para JSONL próprio
├── treinar_tipos.py            # treino conjunto dos três tipos
├── verificar_checkpoint.py     # replay da validação após recarregar os pesos
├── decidir_json.py             # inferência PyTorch
├── avaliar_tipos.py            # avaliação PyTorch, DecisionLFM ou CLEF
├── metricas.py                 # definições compartilhadas das métricas
├── comparar.py                 # comparação pareada e faixas de tokens
├── exemplos/                   # entrada JSON e revisões dos datasets
├── gguf/                       # exportação, backend CUDA e avaliação GGUF
└── tests/                      # testes de entrada, loss, pooling e métricas
```

Pesos, dados, binários, logs e resultados ficam em diretórios ignorados pelo Git. Não há caminhos de usuário ou de servidor fixados nos comandos Python.

## Instalação

Comandos a partir desta pasta. Use Python 3.12 ou 3.13; o ambiente Linux dos experimentos usa Python 3.13.

```bash
cd DecisionLFM/DecisioLFM-Brier-Input
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O treino e a inferência PyTorch usam CUDA com BF16 ou Apple Metal/MPS. O caminho GGUF documentado abaixo usa Linux/CUDA. A disponibilidade de MPS no código não garante que o treino completo caiba na memória do Mac.

Baixe o modelo-base uma vez; ele também é necessário para recarregar os adaptadores LoRA:

```bash
hf download LiquidAI/LFM3-VL-2B \
  --revision f83d2b0a2296aaa277f394990a9b8b949ddf9691
```

Depois dos downloads, `HF_HUB_OFFLINE=1` permite usar o cache sem consultar a rede.

## Entrada e respostas

O exemplo completo está em [`exemplos/entrada.json`](exemplos/entrada.json):

```json
{
  "state": {"mensagem": "Meu cartão foi recusado e preciso pagar uma emergência."},
  "questions": {
    "intencao": {
      "type": "choice",
      "instructions": "Qual problema o cliente relata?",
      "criteria": {"cartao": "Pagamento com cartão recusado", "saque": "Problema ao sacar dinheiro"}
    },
    "urgencia": {
      "type": "noul",
      "instructions": "O cliente precisa resolver o problema com urgência?"
    },
    "gravidade": {
      "type": "score",
      "instructions": "Qual é a gravidade do problema?",
      "criteria": ["Baixa", "Moderada", "Alta", "Crítica"]
    }
  }
}
```

- `state` aceita texto ou outro valor JSON. Valores estruturados são serializados de forma determinística.
- Cada ID em `questions` identifica uma pergunta. Se `instructions` estiver ausente ou vazia, o próprio ID serve como instrução.
- `choice` requer um objeto `criteria` com ao menos duas alternativas `{id: descrição}`. A resposta contém `choice`, `confidence` e `probabilities`.
- `noul` retorna `noul = P(true)`, entre 0 e 1. Opcionalmente, `criteria` pode substituir as descrições de `true` e `false`.
- `score` requer uma lista ordenada com ao menos dois níveis. A resposta contém `score = soma(i × P(i))`, com índices de `0` a `N−1`, além de `probabilities` e `legend`.

`confidence` é a maior probabilidade, como no scorer original. A interface segue o contrato de decisões tipadas usado pelo CLEF; suas fórmulas de confiança e o processamento conjunto de campos não foram copiados. O DecisionLFM percorre as perguntas sequencialmente, e `usage.input_tokens` soma os tokens de todos esses prompts. `usage.output_tokens` é zero.

Inferência PyTorch com um checkpoint treinado:

```bash
python decidir_json.py --checkpoint modelos/r8 \
  --input exemplos/entrada.json
```

Checkpoints anteriores com `decision_config.json`, `head.safetensors`, `tokenizer/` e, quando usado, `adapter/` continuam compatíveis. Você pode fornecer diretamente o caminho de um checkpoint existente em `--checkpoint`; reorganizar esta pasta não altera os pesos.

## Preparar dados

### Benchmark dos experimentos

As revisões das nove fontes estão em `exemplos/fontes_benchmark.json`:

```bash
python dados_tipos.py --output-dir dados/multitarefa \
  --sources exemplos/fontes_benchmark.json --seed 42
```

A configuração usada nos experimentos contém 12.000 exemplos de treino, 1.800 de validação e 3.874 de teste. BANKING77, AG News, BoolQ, SNLI, SST-5 e Yelp entram no treino; Emotion, RTE e Amazon são tarefas reservadas para teste. Cada split possui grupos de documentos separados. Os exemplos acima do limite padrão de 1.024 tokens são excluídos durante a preparação, sem truncamento.

Sem `--sources`, o preparador resolve as revisões disponíveis no momento e as registra no manifesto. Use as revisões fornecidas para reproduzir o benchmark histórico.

### JSONL próprio

Crie `train.jsonl`, `validation.jsonl` e `test.jsonl`. Cada linha de treino/avaliação tem **uma pergunta**, seu alvo e os metadados do exemplo:

```json
{"id":"treino_001","task":"atendimento","type":"choice","held_out_task":false,"group":"documento_001","state":{"mensagem":"Meu cartão foi recusado"},"questions":{"intencao":{"type":"choice","instructions":"Qual problema foi relatado?","criteria":{"cartao":"Pagamento recusado","saque":"Falha no saque"}}},"targets":{"intencao":"cartao"}}
```

Use IDs únicos. `group` identifica o documento de origem e deve ser diferente entre splits; perguntas sobre o mesmo documento pertencem ao mesmo grupo. `held_out_task=true` é permitido somente no teste. Os alvos são IDs de alternativas para `choice`, `"true"`/`"false"` para `noul` e índices como `"0"`, `"1"` para `score`.

O treino conjunto requer exemplos dos três tipos no treino e na validação. A inferência JSON aceita múltiplas perguntas; a supervisão e o benchmark usam uma pergunta por registro.

```bash
python manifesto.py --data-dir dados/meus-dados --max-length 1024
```

Esse comando valida o contrato e cria `manifest.json` com contagens e hashes. Ele não tokeniza nem corta os exemplos: o treino recusa entradas que excedam o limite registrado.

## Treinar e verificar

```bash
python treinar_tipos.py --data-dir dados/multitarefa --output-dir modelos/r8 \
  --epochs 3 --lora-rank 8 --brier-weight 1.0 \
  --batch-size 4 --gradient-accumulation 4 \
  --head-lr 0.001 --lora-lr 0.0001 --seed 42

python verificar_checkpoint.py --checkpoint modelos/r8 \
  --data-dir dados/multitarefa --output resultados/replay-r8.json
```

Padrões: rank 8, alpha 16, dropout LoRA 0,05 e batch efetivo 16. O backbone-base fica congelado; a cabeça e os adaptadores LoRA são atualizados em todas as três épocas. O checkpoint final é salvo após a última época, sem escolher pesos pelo teste.

`training.json` registra a função objetivo, revisões, hashes, hiperparâmetros e histórico. Checkpoints intermediários e estado do otimizador são salvos a cada 100 atualizações; `--checkpoint-every 0` desativa isso. Cada treino exige um diretório de saída novo. Para conferir um treino concluído, o replay recarrega os pesos e compara acurácia e loss da validação com o último histórico; use o mesmo backend/precisão do treino.

## Avaliar em PyTorch

```bash
python avaliar_tipos.py --data-dir dados/multitarefa \
  --checkpoint modelos/r8 --output-dir resultados/decision-bf16
```

Para avaliar o CLEF original no mesmo conjunto, baixe seu snapshot completo primeiro:

```bash
hf download Cloudflare/clef-flash \
  --revision f0b7d3cd65296abfe459e6c228b618f5d8b26d2f \
  --local-dir modelos/clef-flash-hf
python avaliar_tipos.py --data-dir dados/multitarefa --clef \
  --clef-dir modelos/clef-flash-hf --output-dir resultados/clef-bf16
```

A mesma pasta do snapshot é usada pelo avaliador PyTorch e pela conversão GGUF. Sem `--clef-dir`, o avaliador procura a revisão fixada no cache do Hugging Face. O CLEF PyTorch usa os kernels disponíveis no ambiente; o protocolo e as versões dos pacotes ficam em `run.json`.

## GGUF F16, Q8 e Q4

### Compilar o backend

Requer Git, CMake, compilador C++ e CUDA Toolkit com `nvcc`. O script fixa o llama.cpp no commit `0ec4027a2a0183de87c007069adddc06e665f7de` e aplica o patch desta pasta para extrair hidden states do LFM3 sem a projeção gerativa de vocabulário.

```bash
python -m pip install -r gguf/requirements.txt
bash gguf/preparar_backend.sh
```

Por padrão, o checkout fica em `backends/llama.cpp` e o build em `backends/llama.cpp/build-decisionlfm`. Também é possível fornecer caminhos existentes:

```bash
CUDA_ARCHITECTURES=89 JOBS=4 bash gguf/preparar_backend.sh \
  /caminho/llama.cpp /caminho/llama.cpp/build-decisionlfm
```

A revisão deve corresponder à fixada. O script preserva checkouts de outra revisão e informa o conflito. A biblioteca compilada fica em `gguf/libdecisionlfm.so`. A arquitetura `89` corresponde à RTX 4090; sem a variável, o build usa `native`.

### Exportar

```bash
python gguf/exportar.py --checkpoint modelos/r8 \
  --output-dir modelos/gguf-decision --quants q8_0 q4_0
```

O exportador funde a LoRA em FP32 na CPU, converte o backbone textual para F16 e quantiza a partir dele. A cabeça original é exportada diretamente de seus pesos FP32 para cada formato. Saídas: `backbone-f16.gguf`, `head-f16.gguf` e os pares Q8_0/Q4_0; `export.json` registra tipos de tensores, tamanhos e hashes. Normas, vieses e tensores não elegíveis continuam em ponto flutuante. `--quants f16` produz apenas F16.

Para o CLEF, o conversor inclui sua cabeça nativa no próprio arquivo:

```bash
python gguf/exportar.py --clef-dir modelos/clef-flash-hf \
  --output-dir modelos/gguf-clef --quants q8_0 q4_0
```

Com um checkout/build externo, acrescente `--llama-dir /caminho/llama.cpp --build-dir /caminho/build` aos comandos de exportação. Os arquivos GGUF são executados em GGML; não são carregados pelo Transformers para inferência desquantizada.

### Inferência JSON em GGUF

```bash
python gguf/decidir_json.py \
  --backbone modelos/gguf-decision/backbone-q4_0.gguf \
  --head modelos/gguf-decision/head-q4_0.gguf \
  --tokenizer modelos/r8/tokenizer --input exemplos/entrada.json
```

O DecisionLFM usa backbone em CUDA e a cabeça cosseno em GGML/CPU com oito threads. O CLEF usa backbone e cabeça nativos no grafo CUDA. A tokenização continua usando os encoders originais, preservando os IDs de tokens e spans usados na avaliação PyTorch.

O contexto GGUF padrão é 2.048 tokens por pergunta; `--capacity` permite alterá-lo. Entradas excedentes são recusadas, sem truncamento. Uma biblioteca externa pode ser fornecida por `--library`.

### Avaliação e comparação

```bash
python gguf/avaliar.py --family decision --quantization Q4_0 \
  --backbone modelos/gguf-decision/backbone-q4_0.gguf \
  --head modelos/gguf-decision/head-q4_0.gguf --tokenizer modelos/r8/tokenizer \
  --data-dir dados/multitarefa --output resultados/decision-q4

python gguf/avaliar.py --family clef --quantization Q4_0 \
  --backbone modelos/gguf-clef/backbone-q4_0.gguf --clef-dir modelos/clef-flash-hf \
  --data-dir dados/multitarefa --output resultados/clef-q4

python comparar.py --clef resultados/clef-q4 --decision resultados/decision-q4 \
  --output-dir resultados/comparacao-q4
```

Para Q8, use os arquivos `q8_0`, `--quantization Q8_0` e diretórios de resultado novos. Para o controle F16, use os arquivos `f16` e `--quantization F16`. `--limit-per-task N` no avaliador GGUF permite uma verificação pequena antes da avaliação completa.

O comparador também aceita as duas avaliações PyTorch, mas exige o mesmo hardware, protocolo e contador de memória. Ele gera `COMPARACAO.md` e `comparison.json`, incluindo faixas pareadas de tokens. O wrapper GGUF do CLEF usado no benchmark requer uma pergunta por registro; a inferência JSON do DecisionLFM aceita várias perguntas sequenciais.

## Definições das métricas

- Acurácia usa argmax; falhas contam como erro.
- Macro-F1 agregado é a média sem pesos do Macro-F1 de cada tarefa.
- Brier é a soma sobre todas as alternativas válidas, na escala `[0,2]`; falhas são excluídas das métricas de probabilidade.
- `score` usa argmax para acurácia/F1 e valor esperado para MAE.
- Latência inclui preparação JSON, tokenização, transferências, backbone, cabeça, probabilidades e sincronização. Exclui carregamento, aquecimento e escrita.
- VRAM PyTorch é o pico de alocação de tensores, incluindo os pesos. No GGUF, a tabela usa o máximo observado da memória do processo via NVML, amostrado a cada 2 ms, incluindo contexto CUDA. Os contadores são diferentes.
- MPS usa memória unificada observada, sem chamá-la de VRAM dedicada ou pico interno da chamada.
- `usage.input_tokens` conta os prompts efetivamente processados; `output_tokens=0`. As faixas de comparação são definidas pelo maior prompt de cada par, mantendo os mesmos IDs nos dois modelos. Tokenizadores e formatos próprios produzem comprimentos diferentes.

## Resultados históricos

Estes resultados são dos testes anteriores à organização desta pasta: RTX 4090, mesmos 3.874 exemplos, batch 1 e checkpoint DecisionLFM CE+Brier/rank 8/três épocas. São referência documental; não são uma nova medição desta reorganização.

| Métrica | CLEF Q8 | Decision Q8 | CLEF Q4 | Decision Q4 |
|---|---:|---:|---:|---:|
| Acurácia | 77,41% | 71,66% | 77,16% | 70,91% |
| Macro-F1 | 75,85% | 68,39% | 75,55% | 67,81% |
| Brier ↓ | 0,322330 | 0,420488 | 0,323776 | 0,422204 |
| Latência mediana | 25,42 ms | 6,30 ms | 24,60 ms | 6,08 ms |
| Latência p95 | 160,36 ms | 15,14 ms | 166,10 ms | 15,08 ms |
| VRAM observada NVML | 8,92 GiB | 2,46 GiB | 5,17 GiB | 1,65 GiB |
| Falhas | 0 | 0 | 0 | 0 |
| Tokens de entrada, mediana | 225 | 104 | 225 | 104 |

Q8/Q4 representam a precisão dos pesos quantizáveis. Memória de execução também inclui contexto, caches e buffers; por isso ela não cai exatamente pela metade ou para um quarto. A comparação de velocidade considera pipelines e prompts nativos distintos.

Execuções históricas preservadas no servidor: `ce-brier-r8-3ep-ggufq8-20261031T140440Z` e `ggufq4-r8-20261031T152802Z`. Código-base: commit `ac1b21a`; modelo-base: revisão `f83d2b0a2296aaa277f394990a9b8b949ddf9691`; CLEF: revisão `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`.

## Verificação do código

```bash
python -m unittest discover -s tests -v
```

Os testes carregam apenas o tokenizer, sem os pesos do backbone. Se ele já estiver disponível em uma pasta local:

```bash
HF_HUB_OFFLINE=1 DECISIONLFM_TEST_TOKENIZER=/caminho/checkpoint/tokenizer \
  python -m unittest discover -s tests -v
```

Os checks de integração podem usar os comandos de replay, inferência e avaliação limitada acima com um checkpoint existente. Reorganizar o código não exige novo treino para conferir a compatibilidade dos pesos.

### Conferência desta organização

Foram executados 21 testes automatizados, replay dos 1.800 exemplos de validação com diferença de loss igual a zero, preparação reduzida das nove fontes e inferência do JSON com três perguntas em PyTorch e GGUF. A exportação produziu F16/Q8/Q4 com os mesmos tensores da execução anterior.

Na verificação de integração, nove exemplos por modelo e quantização Q8/Q4 tiveram os mesmos tokens, previsões e probabilidades dos resultados históricos (36 decisões verificadas, diferença máxima de probabilidade zero). Essa verificação é uma amostra de compatibilidade; a tabela histórica acima continua sendo a avaliação completa de 3.874 exemplos.
