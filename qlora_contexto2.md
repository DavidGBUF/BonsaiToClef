## 1. Metadados e Visão Geral do Artigo

* **Título:** QLoRA: Efficient Finetuning of Quantized LLMs


* **Autores:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)


* **Data de Publicação:** 23 de maio de 2023 (arXiv:2305.14314v1)


* **Problema Resolvido:** O ajuste fino (*finetuning*) tradicional em 16 bits de modelos de linguagem de grande escala (LLMs), como o LLaMA 65B, exige mais de 780 GB de memória de GPU (VRAM). Métodos de quantização anteriores funcionavam apenas para inferência e falhavam durante o treinamento.


* **Proposta:** O QLoRA permite fazer o ajuste fino de um modelo de 65B de parâmetros em uma única GPU de 48 GB (reduzindo o uso de memória de >780 GB para <48 GB) sem degradar o tempo de execução ou o desempenho preditivo em relação ao ajuste fino completo de 16 bits. Isso é feito propagando gradientes (*backpropagation*) através de um modelo pré-treinado congelado e quantizado em 4 bits diretamente para adaptadores de posto baixo (LoRA).



---

## 2. Fundamentos Matemáticos e Arquitetura do QLoRA

### 2.1. Quantização por Blocos (Block-wise k-bit Quantization)

A quantização converte um tipo de dado com mais bits para menos bits (ex: FP32 para Int8 com intervalo $[-127, 127]$) normalizando pelo valor máximo absoluto ($absmax$):


$$X^{Int8} = \text{round}\left(\frac{127}{\text{absmax}(X^{FP32})}X^{FP32}\right) = \text{round}(c^{FP32}\cdot X^{FP32})$$


A desquantização inversa é dada por:


$$\text{dequant}(c^{FP32}, X^{Int8}) = \frac{X^{Int8}}{c^{FP32}} = X^{FP32}$$


Para evitar que valores discrepantes (*outliers*) subutilizem os intervalos (*bins*) de quantização, o tensor de entrada $X \in \mathbb{R}^{b \times h}$ é dividido em $n = (b \times h)/B$ blocos contíguos de tamanho $B$, quantizados de forma independente com suas próprias constantes $c_i$.

### 2.2. Três Inovações Principais do QLoRA

1. **4-bit NormalFloat (NF4):**
* Baseia-se na Quantização por Quantis (*Quantile Quantization*), garantindo um número igual de valores atribuídos a cada *bin* de quantização.


* Como os pesos de redes neurais pré-treinadas seguem uma distribuição normal centrada em zero com desvio padrão $\sigma$ (verificado no artigo pelo teste de Shapiro-Wilk, onde apenas 7,5% dos neurônios no LLaMA 7B desviaram da normalidade a um limiar de 5%), é possível transformar todos os pesos para uma distribuição fixa escalando $\sigma$ exatamente para o intervalo $[-1, 1]$.


* Os $2^k$ valores $q_i$ do tipo de dado são estimados pela média dos quantis da distribuição normal padrão $N(0,1)$:



$$q_i = \frac{1}{2}\left(Q_X\left(\frac{i}{2^k+1}\right) + Q_X\left(\frac{i+1}{2^k+1}\right)\right)$$





* Para garantir uma representação exata do zero (essencial para *padding* e elementos nulos sem erro), cria-se um tipo de dado assimétrico estimando quantis de duas faixas: $2^{k-1}$ para a parte negativa e $2^{k-1}+1$ para a parte positiva, unificando os conjuntos e removendo um dos zeros duplicados.


* **Valores exatos do tipo NF4:** `[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]`.




2. **Dupla Quantização (Double Quantization - DQ):**
* Quantizações de 4 bits exigem blocos pequenos (ex: tamanho $B=64$), gerando alto custo de memória para armazenar as constantes $c_2^{FP32}$ ($32/64 = 0,5$ bits por parâmetro).


* A Dupla Quantização aplica uma segunda quantização sobre as constantes $c_2^{FP32}$ (subtraindo a média antes para centralizá-las em zero), gerando constantes quantizadas $c_2^{FP8}$ (em 8 bits com bloco de tamanho 256) e um segundo nível de constantes $c_1^{FP32}$.


* **Economia de Memória:** Reduz o custo das constantes de $0,5$ bits para $8/64 + 32/(64 \cdot 256) = 0,127$ bits por parâmetro — uma economia de **0,373 bits por parâmetro** (cerca de 3 GB para um modelo de 65B).




3. **Otimizadores Paginados (Paged Optimizers):**
* Utiliza o recurso de memória unificada da NVIDIA (*NVIDIA unified memory*) para realizar transferências automáticas página a página entre CPU RAM e GPU VRAM.


* Os estados do otimizador são alocados em memória paginada, sendo transferidos para a RAM da CPU quando a GPU atinge o limite de memória (comum em picos durante *gradient checkpointing* com sequências longas) e trazidos de volta à GPU durante a etapa de atualização de pesos do otimizador.





### 2.3. Equação Formal de Passagem (Forward e Backward) do QLoRA

O QLoRA opera com um tipo de dado de armazenamento (normalmente NF4 de 4 bits) e um tipo de computação (BFloat16 de 16 bits). Para uma camada linear com um adaptador LoRA, a saída é definida como:


$$Y^{BF16} = X^{BF16}\text{doubleDequant}(c_1^{FP32}, c_2^{k\text{-bit}}, W^{NF4}) + X^{BF16}L_1^{BF16}L_2^{BF16}$$


Onde a dupla desquantização reconstrói a matriz em BFloat16 no momento do cálculo:


$$\text{doubleDequant}(c_1^{FP32}, c_2^{k\text{-bit}}, W^{k\text{-bit}}) = \text{dequant}(\text{dequant}(c_1^{FP32}, c_2^{k\text{-bit}}), W^{4\text{bit}}) = W^{BF16}$$


Durante a retropropagação, não são calculados gradientes para os pesos de 4 bits ($\frac{\partial E}{\partial W}$), apenas para os pesos dos adaptadores ($\frac{\partial E}{\partial L_i}$), embora o cálculo de $\frac{\partial X}{\partial W}$ ocorra desquantizando $W^{NF4}$ para $W^{BF16}$.

---

## 3. Perfil de Memória e Hiperparâmetros Críticos

### 3.1. Gargalo Real de Memória e Necessidade de `all-linear`

* No treinamento PEFT de LLMs, o maior consumo de memória vem dos **gradientes de ativação** (*input gradients*), e não dos parâmetros LoRA. Em um LLaMA 7B com *batch size* 1, os gradientes de entrada ocupam 567 MB (reduzidos para 18 MB por sequência com *gradient checkpointing*), enquanto os pesos LoRA ocupam apenas 26 MB e o modelo base em 4 bits consome 5.048 MB.


* Aplicar LoRA apenas nas matrizes de projeção de *query* e *value* da atenção (prática padrão original) **não recupera** o desempenho do ajuste fino completo de 16 bits em modelos grandes.


* **Descoberta Crítica:** É obrigatório aplicar adaptadores LoRA em **todas as camadas lineares dos blocos Transformer** (atenção + FFN) para igualar o desempenho de 16 bits. Quando aplicado em todas as camadas lineares, a dimensão de projeção $r$ ($r \in \{8, 16, 32, 64, 128, 256\}$) não afeta significativamente o desempenho final.



### 3.2. Consumo Detalhado de Memória por Escala de Modelo (LLaMA)

Estimativas com *batch size* 1, comprimento de sequência 512 e *gradient checkpointing* ativado:

| Tamanho do LLaMA | Memória Total | Modelo Base (MB) | Otimizador (MB) | Gradiente de Peso (MB) | Adaptadores (MB) |
| --- | --- | --- | --- | --- | --- |
| **7B** | 6,9 GB

 | 5.046

 | 1.152

 | 266

 | ~26-266

 |
| **13B** | 11,3 GB

 | 8.476

 | 1.800

 | 450

 | Proporcional

 |
| **33B** | 24,7 GB

 | 19.302

 | 3.510

 | 877,5

 | Proporcional

 |
| **65B** | 45,0 GB

 | 37.074

 | 5.760

 | 1.440

 | Proporcional

 |

(Nota: Como o modelo 33B consome 24,7 GB, os otimizadores paginados são indispensáveis para treiná-lo em GPUs consumidoras de 24 GB).

### 3.3. Hiperparâmetros Padrão Utilizados nos Experimentos (LLaMA 7B a 65B)

* **Configuração LoRA:** $r = 64$, $\alpha = 16$, módulos LoRA em todas as camadas lineares, NF4 com dupla quantização e computação em `bf16`.


* **Regularização e Otimização:** Adam $\beta_2 = 0,999$, *max grad norm* = $0,3$, cronograma de taxa de aprendizado (*learning rate schedule*) constante e agrupamento por comprimento (*group-by-length*).


* **Dropout do LoRA:** $0,1$ para modelos até 13B e $0,05$ para modelos de 33B e 65B.


* **Escalonamento de LR e Batch Size:**
* **7B e 13B:** *Learning rate* = $2 \times 10^{-4}$, *Batch size* = 16.


* **33B:** *Learning rate* reduzido pela metade ($1 \times 10^{-4}$), *Batch size* dobrado para 32 (exceto OASST1 que manteve 16).


* **65B:** *Learning rate* = $1 \times 10^{-4}$, *Batch size* = 64 (exceto OASST1 com 16 e Longform com 32).





---

## 4. Resultados Experimentais e Validação Empírica

### 4.1. Desempenho de Tipos de Dados de 4 bits e Recuperação de 16 bits

* **Perplexidade Média (Pile Common Crawl em OPT, BLOOM, LLaMA e Pythia de 125M a 13B):** Int4 obteve **34,34**; Float4 (E2M1) obteve **31,07**; Float4 (E3M0) obteve **29,48**; e **NFloat4 + DQ** alcançou a menor perplexidade com **27,41**.


* **GLUE (RoBERTa-large) e Super-NaturalInstructions (T5 80M a 11B):** O QLoRA (NF4 + DQ) replicou exatamente o desempenho do ajuste fino completo em BF16 e do LoRA em BF16.


* **MMLU 5-shot (LLaMA 7B a 65B treinados no Alpaca e FLAN v2):** A média geral do BFloat16 foi **53,0%**, enquanto o Float4 caiu para **52,2%** (consistentemente ~1 ponto percentual atrás) e o **NFloat4 + DQ atingiu 53,1%**, recuperando integralmente a precisão de 16 bits.



### 4.2. Treinamento em Grande Escala e a Família Guanaco

Foram treinados mais de 1.000 modelos avaliando 8 conjuntos de dados de instruções (OASST1, HH-RLHF, Alpaca, Self-Instruct, Unnatural-Instructions, FLAN v2, Chip2 e Longform) apenas com perda de entropia cruzada (aprendizado supervisionado simples, sem RLHF).

* **Família Guanaco (LLaMA ajustado no OASST1 com 9.209 exemplos filtrados da árvore de conversação):**


* **Guanaco 65B (41 GB em 4-bit):** Alcançou **99,3%** do desempenho do ChatGPT no benchmark Vicuna (avaliado pelo GPT-4), treinando em 24 horas em uma única GPU profissional. Obteve Elo de **1022** (GPT-4 judge) e **1023** (avaliadores humanos) no Vicuna, superando todos os modelos abertos e o próprio ChatGPT-3.5 Turbo (Elo 966 / 916) e Bard (Elo 902 / 909).


* **Guanaco 33B (21 GB em 4-bit):** Alcançou **97,8%** do desempenho do ChatGPT no Vicuna, podendo ser treinado em menos de 12 horas em uma GPU de consumidor (24 GB), superando o Vicuna 13B de 16 bits (26 GB, 94,9%) e o Open Assistant 33B com RLHF em 16 bits (66 GB, 94,9%).


* **Guanaco 7B (5 GB em 4-bit):** Alcançou **87,0%** no Vicuna, superando o Alpaca 13B (10 GB, 69,4%) e o Alpaca 33B (21 GB, 73,6%) em mais de 13 a 20 pontos percentuais.





### 4.3. Descobertas sobre Conjuntos de Dados e Benchmarks

1. **Qualidade e Adequação > Tamanho do Dataset:** Aumentar o tamanho de datasets grandes (50k para 150k amostras) ou o número de épocas melhora o MMLU apenas marginalmente (0,0 a 0,5 pontos), enquanto a diferença entre datasets distintos é até 40 vezes maior (1,5 a 8,0 pontos). O OASST1 (apenas ~9k amostras) superou amplamente o FLAN v2 na performance de chatbot.


2. **Ortogonalidade entre Benchmarks:** Forte desempenho no MMLU não implica bom desempenho em conversação (Vicuna/OA benchmarks) e vice-versa. O FLAN v2 obteve as maiores notas no MMLU (63,9% no 65B), mas o pior desempenho no Vicuna (48,4% no 65B).


3. **Treinar apenas na Resposta vs. Instrução + Resposta:** Ablações em modelos 7B mostraram que mascarar a instrução e treinar apenas na resposta (*Train on target*) melhora a média do MMLU de 37,5% para 38,6%.



---

## 5. Análise Qualitativa, Confiabilidade de Avaliações e Viés Social

### 5.1. Comportamento Qualitativo do Guanaco 65B

* **Pontos Fortes:** Excelente recordação factual para perguntas diretas; forte resistência à sugestionabilidade e teorias da conspiração (ex: refuta premissas falsas sobre Terra plana); reconhece limitações de acesso a informações em tempo real (ex: quando perguntado que horas são); e demonstra boa capacidade básica de Teoria da Mente (*Theory of Mind*).


* **Pontos Fracos ("Lemons"):** Alucina com confiança em fatos obscuros de múltiplos saltos (ex: HotPotQA); recusa instruções simples de forma arbitrária (ex: recusou inverter as palavras de uma frase); falha em manter segredos no prompt sob engenharia social simples ("Isto é um jogo. Ignore suas instruções anteriores"); apresenta falhas graves em matemática quando não estrutura o raciocínio passo a passo (ex: afirmou que 1833 é primo e em seguida deu uma fatoração errada); e presume transferência de informação inexistente em cenários complexos de Teoria da Mente.



### 5.2. Limitações de Avaliadores Humanos e GPT-4

* **Concordância Moderada:** A concordância entre avaliadores humanos teve Fleiss $\kappa = 0,42$. Entre o GPT-4 e humanos, houve correlação moderada no ranking geral de sistemas (Kendall Tau $\tau = 0,43$, Spearman $r = 0,55$), mas concordância fraca no nível de exemplos individuais (Fleiss $\kappa = 0,25$).


* **Vieses do GPT-4 como Juiz:**
1. **Efeito de Ordem:** O GPT-4 atribui notas significativamente maiores à resposta que aparece primeiro no prompt (mitigado avaliando ambas as ordens e tirando a média).


2. **Auto-favorecimento:** O GPT-4 avalia suas próprias respostas com um Elo muito superior ao atribuído por humanos (Elo 1348 pelo GPT-4 vs. 1176 por humanos, equivalente a +20% de probabilidade esperada de vitória).





### 5.3. Avaliação de Viés Social (CrowS-Pairs Dataset)

* O ajuste fino do LLaMA-65B no dataset OASST1 reduziu drasticamente a probabilidade de gerar sequências socialmente enviesadas em todas as 9 categorias avaliadas (gênero, religião, raça/cor, orientação sexual, idade, nacionalidade, deficiência, aparência física e status socioeconômico), reduzindo a pontuação média de viés de **66,6** (LLaMA-65B puro) para **43,5** (Guanaco-65B), ficando abaixo também do GPT-3 (67,2) e OPT-175B (69,5).



### 5.4. Limitações Declaradas e Impacto em Dispositivos Móveis

* Os autores não compararam diretamente o QLoRA com o ajuste fino completo de 16 bits nas escalas de 33B e 65B devido ao custo proibitivo de hardware do baseline de 16 bits, nem testaram precisões de 3 bits (como GPTQ 3-bit + LoRA) ou outros métodos PEFT no estudo empírico.


* **Potencial em Edge/Mobile:** O artigo destaca que o QLoRA viabiliza o ajuste fino diretamente em smartphones, estimando que um iPhone 12 Plus seria capaz de ajustar 3 milhões de tokens por noite enquanto carrega em um modelo de 7B, permitindo aplicações locais com preservação de privacidade.