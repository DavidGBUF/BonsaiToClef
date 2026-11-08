# BonsaiToClef

Experimento simples de inferência zero-shot do **CLEF-Flash** no **BANKING77**.
Sem treinamento, exemplos rotulados no prompt ou alteração dos pesos.

- `inferir.py`: executa uma mensagem por vez e salva as previsões.
- `avaliar.py`: calcula métricas, sem carregar o modelo e sem usar GPU.

## Instalação

Ambiente inicial: Linux, Python 3.13 e RTX 4090 de 24 GB. A inferência requer
CUDA e suporte a BF16. Os pesos completos ocupam aproximadamente 19,1 GB no
download; a execução também usa memória para ativações e operações intermediárias.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O primeiro uso baixa o modelo e o dataset para o cache padrão do Hugging Face.
Os downloads são públicos e anônimos; não é necessário configurar um token.
O carregador Python publicado pela Cloudflare é importado do snapshot do modelo.
As revisões padrão do modelo e do dataset estão fixadas no código e ficam
registradas em cada execução. O processador oficial requer Pillow e torchvision,
mesmo quando o experimento utiliza somente texto.

## Primeiro teste: 20 exemplos de desenvolvimento

```bash
python inferir.py --split train --limit 20 --seed 42 --output-dir resultados/train-20
python avaliar.py resultados/train-20
```

Esses exemplos servem para conferir a execução. Não representam um resultado
final do benchmark. Qualquer revisão de instruções deve ocorrer nesta etapa.

## Avaliação completa

Após congelar as instruções e alternativas:

```bash
python inferir.py --split test --output-dir resultados/test
python avaliar.py resultados/test
```

O teste oficial contém 3.080 exemplos. Todas as 77 intenções são oferecidas em
cada requisição `choice`. A descrição de cada intenção é seu nome com os
underscores substituídos por espaços e sem `?`; essas descrições não são
ajustadas com exemplos do teste. A resposta é a alternativa de maior probabilidade.
O rótulo verdadeiro é usado apenas na avaliação.

O padrão é BF16, batch 1, limite de 8.192 tokens e três chamadas de aquecimento.
O limite inclui a mensagem, as instruções e as 77 alternativas. Entradas que
ultrapassam o limite são registradas como falha, sem truncamento silencioso.
`--limit 0` (padrão) usa todo o split; um limite positivo seleciona uma amostra
sem reposição com a seed informada. Um diretório de saída existente é recusado.

## Saídas e métricas

Cada diretório em `resultados/` contém:

- `run.json`: revisões do modelo/dataset, hash do script, índices selecionados, esquema,
  parâmetros, GPU, versões das bibliotecas e status da execução.
- `predictions.jsonl`: ID, texto, rótulo esperado, previsão, probabilidades,
  latência, número de tokens, pico de memória alocada e eventual erro.
- `metrics.json`: acurácia, macro-F1, Brier multiclasses, latência mediana/p95,
  pico de VRAM alocada, falhas e pares de confusão mais frequentes.

Acurácia e macro-F1 incluem falhas como previsões incorretas; a macro-F1 usa
todas as 77 classes. Brier e latência consideram somente previsões válidas,
com a quantidade de registros usada explicitamente informada.
O Brier é a média por exemplo da soma dos erros quadráticos das 77
probabilidades contra o rótulo one-hot, sem dividir pelo número de classes
(intervalo de 0 a 2; menor é melhor).

A latência inclui codificação, transferência dos tensores, inferência e obtenção
das probabilidades, com sincronização CUDA. Exclui leitura do dataset,
download, carregamento, aquecimento e escrita dos resultados.
O pico informado é `torch.cuda.max_memory_allocated`, incluindo os pesos;
não é o consumo total mostrado pelo `nvidia-smi`.
Execuções interrompidas preservam as previsões já gravadas, mas o avaliador
recusa calcular métricas finais de uma execução incompleta.

Para reutilizar a configuração de uma execução, use os mesmos argumentos e as revisões
registradas em `run.json`:

```bash
python inferir.py --split test --model-revision SHA_MODELO --dataset-revision SHA_DATASET --output-dir resultados/repeticao
```

Os resultados não reproduzem automaticamente os números da Cloudflare:
o dataset pode ser o mesmo, mas instruções e protocolo precisam coincidir para
uma comparação direta. Latência e memória dependem do hardware e da entrada.

## Validação executada em 2026-10-28

RTX 4090 de 24 GB, CUDA 13.0, BF16, batch 1, implementação PyTorch
oficial. O teste inicial com 20 exemplos do split `train` terminou sem falhas.
A avaliação seguinte utilizou todos os exemplos do split `test`, mantendo
as mesmas instruções e alternativas.

| Métrica | Resultado |
| --- | ---: |
| Exemplos de teste | 3080 |
| Falhas de execução | 0 |
| Acurácia | 95.91% |
| Macro-F1 | 95.90% |
| Brier multiclasses | 0.067443 |
| Latência mediana | 312.77 ms |
| Latência p95 | 313.67 ms |
| Pico de VRAM alocada | 18.16 GiB |

Revisões: modelo `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` e dataset
`1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`. As previsões e métricas completas ficam em
`resultados/test/` (ignorado pelo Git). Os 3.080 IDs foram conferidos e cada
uma das 77 classes possui 40 exemplos. A execução utilizou o caminho PyTorch
de referência, com fallback para operações Torch nos blocos de atenção linear.
O avaliador também passou por verificações com métricas analíticas conhecidas,
falhas de execução e recusa de distribuições inválidas, IDs duplicados e
execuções incompletas.

## DecisionLFM

A pasta [DecisionLFM](DecisionLFM/README.md) contém nosso modelo de decisão
baseado no LFM3-VL-2B, com treino de cabeça + LoRA, salvamento e inferência.
Seu README explica a execução e a comparação usando este mesmo avaliador.
O código e o ambiente do baseline CLEF permanecem independentes.

A abordagem [DecisionLFM CE+Brier com entrada JSON](DecisionLFM/DecisioLFM-Brier-Input/README.md)
inclui `choice`, `noul` e `score`, treino de até três épocas com LoRA rank 8,
seleção por validação e calibração por tipo com proteção do MAE de `score`.
O README contém o fluxo completo para preparar dados, treinar, conferir os pesos,
calibrar, responder a um JSON e avaliar. Várias perguntas são respondidas
sequencialmente, com uma passagem pelo backbone por pergunta.

A pasta [DecisionLFM-QLoRA](DecisionLFM/DecisionLFM-QLoRA/README.md) implementa
o treinamento e a inferência com **QLoRA (4-bit NormalFloat4 + Dupla Quantização +
computação BF16)** no módulo de linguagem do LFM3-VL-2B, mantendo a `DecisionHead`
em FP32, suporte a otimizadores paginados (`PagedAdamW32bit` / `PagedAdamW8bit`),
fusão limpa sobre o modelo-base FP32 para exportação GGUF e comparador pareado
direto contra o LoRA BF16 (`comparar_qlora.py`).

## Referências

- [CLEF-Flash e implementação oficial](https://huggingface.co/Cloudflare/clef-flash)
- [BANKING77: 77 intenções, textos em inglês e splits oficiais](https://huggingface.co/datasets/PolyAI/banking77)
