# DecisionLFM-QLoRA — QLoRA (4-bit NF4 + Double Quantization), CE+Brier e Entrada JSON

Esta pasta contém a implementação **DecisionLFM-QLoRA**, independente de `DecisionLFM/DecisioLFM-Brier-Input/`. Ela adapta o backbone **LiquidAI/LFM3-VL-2B** usando **QLoRA (4-bit NormalFloat4 + Dupla Quantização + computação em BFloat16)** no módulo de linguagem, mantendo a cabeça cosseno compartilhada (`DecisionHead`) em FP32 e retornando decisões tipadas (`choice`, `noul` e `score`) em uma única passagem pelo backbone por pergunta, sem gerar texto.

## 1. Arquitetura QLoRA e Decisões de Engenharia

1. **Quantização 4-bit NormalFloat (`nf4`) com Dupla Quantização (`bitsandbytes`)**:
   - O backbone é carregado via `BitsAndBytesConfig`:
     - `load_in_4bit=True`
     - `bnb_4bit_quant_type="nf4"` (quantis exatos da distribuição normal $N(0,1)$)
     - `bnb_4bit_use_double_quant=True` (quantização de segunda ordem sobre as constantes de bloco)
     - `bnb_4bit_compute_dtype=torch.bfloat16` (desquantização sob demanda no *forward* e *backward* para BF16)
     - `llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]` (preserva a torre de visão inativa fora da quantização 4-bit NF4).
2. **Estabilização k-bit (`prepare_model_for_kbit_training`)**:
   - Antes de acoplar os adaptadores LoRA com `get_peft_model()`, `DecisionLFM.add_lora()` executa `prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})`.
3. **Alvos LoRA e Cabeça FP32**:
   - Os adaptadores LoRA são aplicados a todas as projeções lineares do módulo de linguagem (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `w1`, `w2`, `w3`).
   - A `DecisionHead` permanece integralmente em **FP32**.
4. **Otimizadores (`--optimizer`)**:
   - `adamw` (padrão PyTorch, permitindo comparação controlada isolando apenas a quantização 4-bit NF4 do backbone em relação ao `DecisioLFM-Brier-Input`);
   - `paged_adamw_32bit` e `paged_adamw_8bit` (otimizadores paginados do `bitsandbytes` que usam memória unificada CPU/GPU para evitar picos de OOM).
5. **Exportação GGUF sem Dupla Perda de Quantização**:
   - Ao exportar um checkpoint QLoRA para GGUF (`F16`, `Q8_0`, `Q4_0`), `gguf/exportar.py` carrega o modelo-base original (`LiquidAI/LFM3-VL-2B`) em **FP32 na CPU** e funde os adaptadores QLoRA treinados via `merge_and_unload(safe_merge=True)`, evitando acumular ruído de `NF4 -> FP16 -> Q8_0/Q4_0`.

---

## 2. Instalação e Preparação do Ambiente

Use um ambiente virtual próprio dentro desta pasta (Python 3.12 ou 3.13 com CUDA/BF16):

```bash
cd DecisionLFM/DecisionLFM-QLoRA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Baixe o modelo-base na revisão fixada (caso ainda não esteja no cache local):

```bash
hf download LiquidAI/LFM3-VL-2B \
  --revision f83d2b0a2296aaa277f394990a9b8b949ddf9691
```

---

## 3. Fluxo Completo de Execução (PyTorch QLoRA)

Você pode reutilizar o diretório de dados já preparado em `../DecisioLFM-Brier-Input/dados/multitarefa` ou preparar um novo com `dados_tipos.py`:

```bash
# 1. (Opcional) Preparar os 9 datasets do benchmark multitarefa
python dados_tipos.py --output-dir dados/multitarefa \
  --sources exemplos/fontes_benchmark.json --seed 42

# 2. Treinar com QLoRA 4-bit NF4 + Double Quant (até 3 épocas, LoRA rank 8)
python treinar_tipos.py --data-dir dados/multitarefa --output-dir modelos/qlora-r8 \
  --epochs 3 --lora-rank 8 --brier-weight 1.0 --score-brier-weight 2.0 \
  --head-lr 0.0003 --lora-lr 0.00003 --batch-size 4 --gradient-accumulation 4 \
  --optimizer adamw

# Para usar Otimizador Paginado (PagedAdamW32bit) e clip 0.3 do artigo QLoRA:
python treinar_tipos.py --data-dir dados/multitarefa --output-dir modelos/qlora-paged-r8 \
  --epochs 3 --lora-rank 8 --brier-weight 1.0 --score-brier-weight 2.0 \
  --head-lr 0.0003 --lora-lr 0.00003 --batch-size 4 --gradient-accumulation 4 \
  --optimizer paged_adamw_32bit --max-grad-norm 0.3

# 3. Verificar o checkpoint (replay exato da validação em 4-bit NF4)
python verificar_checkpoint.py --checkpoint modelos/qlora-r8 \
  --data-dir dados/multitarefa --output resultados/replay-qlora-r8.json

# 4. Calibrar temperaturas por tipo na metade reservada da validação (protegendo MAE de score)
python calibrar.py --checkpoint modelos/qlora-r8 \
  --data-dir dados/multitarefa --protect-score-mae

# 5. Inferência direta em arquivo JSON com múltiplas perguntas
python decidir_json.py --checkpoint modelos/qlora-r8 --input exemplos/entrada.json

# 6. Avaliar no conjunto de teste em PyTorch (4-bit NF4)
python avaliar_tipos.py --checkpoint modelos/qlora-r8 \
  --data-dir dados/multitarefa --output-dir resultados/teste-qlora-r8
```

---

## 4. Comparação Pareada: LoRA (BF16) vs QLoRA (4-bit NF4)

O script `comparar_qlora.py` compara diretamente uma avaliação do `DecisioLFM-Brier-Input` (LoRA BF16) contra uma avaliação do `DecisionLFM-QLoRA` (4-bit NF4) nos mesmos 3.874 exemplos de teste, auditando também a economia de VRAM e o tempo de treinamento registrados nos respectivos `training.json`:

```bash
python comparar_qlora.py \
  --lora ../DecisioLFM-Brier-Input/resultados/teste-r8 \
  --qlora resultados/teste-qlora-r8 \
  --output-dir resultados/comparacao-lora-vs-qlora
```

Gera `COMPARACAO_QLORA.md` e `comparison-qlora.json` contendo:
- Concordância de decisão (`prediction_agreement_rate`);
- Acurácia, Macro-F1, Brier multiclasses, MAE de `score`, latência mediana/p95 e VRAM de inferência;
- Pico de VRAM alocada no treino (`peak_allocated_vram_gib`) e tempo total de treino;
- Desempenho detalhado por tipo (`choice`, `noul`, `score`) e por tarefa (*in-domain* e *held-out*).

---

## 5. Exportação e Avaliação GGUF (`F16`, `Q8_0`, `Q4_0`)

```bash
# 1. Compilar o backend llama.cpp com o patch LFM3
python -m pip install -r gguf/requirements.txt
bash gguf/preparar_backend.sh

# 2. Exportar fundindo os adaptadores QLoRA sobre o modelo-base FP32 na CPU
python gguf/exportar.py --checkpoint modelos/qlora-r8 \
  --output-dir modelos/gguf-qlora --quants q8_0 q4_0

# 3. Inferência JSON via GGUF
python gguf/decidir_json.py \
  --backbone modelos/gguf-qlora/backbone-q4_0.gguf \
  --head modelos/gguf-qlora/head-q4_0.gguf \
  --tokenizer modelos/qlora-r8/tokenizer --input exemplos/entrada.json

# 4. Avaliar no benchmark via GGUF
python gguf/avaliar.py --family decision --quantization Q4_0 \
  --backbone modelos/gguf-qlora/backbone-q4_0.gguf \
  --head modelos/gguf-qlora/head-q4_0.gguf --tokenizer modelos/qlora-r8/tokenizer \
  --data-dir dados/multitarefa --output resultados/qlora-gguf-q4
```

---

## 6. Testes Automatizados

A suíte possui **37 testes unitários** (os 32 testes originais de tokenização, contrato JSON, perda CE+Brier, calibração e manifesto + 5 novos testes dedicados ao QLoRA em `tests/test_qlora.py`):

```bash
python -m unittest discover -s tests -v
```

