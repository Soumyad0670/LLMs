# 🧠 LLMs — Hands‑on Notebooks for Modern Large Language Models

A lightweight, no‑nonsense collection of Jupyter notebooks exploring core LLM concepts and practical fine‑tuning. If you’re here to *learn by doing* — you’re in the right place.


---

## Table of Contents

* [Project Goals](#project-goals)
* [Repo Structure](#repo-structure)
* [Prerequisites](#prerequisites)
* [Quickstart](#quickstart)
* [Notebook Guides](#notebook-guides)
* [Datasets](#datasets)
* [Hardware Notes](#hardware-notes)
* [Common Pitfalls](#common-pitfalls)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Project Goals

* Build intuition for the Transformer architecture from scratch.
* Fine‑tune popular open and open‑ish models with minimal boilerplate.
* Compare training techniques: full fine‑tuning vs LoRA/QLoRA.
* Keep the code *readable first*, *fast second*.

---

## Repo Structure

```
LLMs/
├── BERT.ipynb                    # Encoder-only model: classification with BERT
├── LLM(transformer).ipynb       # From-scratch Transformer walkthrough (educational)
├── Finetuning_LLama2.ipynb      # Instruction/chat fine-tuning for Llama 2
├── Gemma_tuning_LoRA.ipynb      # Parameter-efficient tuning (LoRA) on Gemma
├── .gitignore
├── LICENSE                      # MIT
└── README.md                    # You are here
```

---

## Prerequisites

* **Python** ≥ 3.10
* **GPU**: NVIDIA CUDA‑capable GPU recommended (>= 12 GB VRAM for most fine‑tunes; 24 GB+ preferred). CPU-only works for the BERT demo, but bring snacks.
* **Key libraries** (installed in Quickstart):

  * `torch`, `transformers`, `datasets`, `accelerate`
  * `peft` (for LoRA/QLoRA), `bitsandbytes` (optional 4‑bit)
  * `scikit-learn`, `evaluate`, `numpy`, `pandas`
  * `trl` (optional, for SFT/RLHF experiments)

> **Model Access:** Some model weights (e.g., *meta-llama/Llama-2-7b-chat-hf*) require access approval on Hugging Face. Request access on the model card with the same HF account that your token uses. After approval, login with `huggingface-cli login` and restart the notebook kernel.

---

## Quickstart

### 1) Create and activate an environment

```bash
# conda (recommended)
conda create -n llms python=3.10 -y
conda activate llms

# or uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# uv venv --python 3.10
# source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA version as needed
pip install transformers datasets accelerate peft bitsandbytes scikit-learn evaluate pandas numpy tqdm sentencepiece safetensors
# optional extras
pip install trl wandb
```

### 3) Authenticate to Hugging Face (if using gated models)

```bash
huggingface-cli login
```

### 4) Run notebooks

Open the notebooks in VS Code (.ipynb) or Jupyter Lab. Execute cells top‑to‑bottom.

---

## Notebook Guides

### 1) `LLM(transformer).ipynb`

**What you’ll learn:**

* Tokenization → embeddings → positional encodings → self‑attention → MLP → logits.
* Causal masking and why your model shouldn’t leak the future.
* Clean, minimal PyTorch blocks (attention, feed‑forward, residuals).

**Outcome:** A *from-scratch* mini‑Transformer you can extend.

---

### 2) `BERT.ipynb`

**Task:** Text classification on GLUE/SST‑2 (or similar) using an encoder‑only architecture.

**Covers:**

* Loading datasets via `datasets.load_dataset`.
* Fine‑tuning `bert-base-uncased` with `Trainer` or `Accelerate`.
* Metrics with `evaluate` (accuracy, F1) and a quick confusion matrix.

**Outcome:** A strong baseline and a mental model for encoder‑only LMs.

---

### 3) `Finetuning_LLama2.ipynb`

**Task:** Supervised fine‑tuning (SFT) for chat/instruction tasks on Llama‑2.

**Covers:**

* Dataset formatting for chat templates (`apply_chat_template`).
* Mixed‑precision (`bf16`/`fp16`), gradient accumulation, and checkpointing.
* Paged optimizers & memory‑savvy configs.

**Notes:** Requires HF access to `meta-llama/*`. Swap in Mistral or Qwen if access is pending.

**Outcome:** A solid SFT run with reproducible training args and eval samples.

---

### 4) `Gemma_tuning_LoRA.ipynb`

**Task:** Parameter‑efficient fine‑tuning (PEFT) with LoRA on Google’s Gemma series.

**Covers:**

* `peft.LoraConfig`, target modules, and rank selection.
* 8‑bit/4‑bit loading via `bitsandbytes` to fit big models on small GPUs.
* Merging LoRA weights back into the base model for export.

**Outcome:** Cheap(‑ish) training that still moves the needle.

---

## Datasets

* **SST‑2 (GLUE)** for classification demos.
* Example instruction datasets (Alpaca‑style, OpenHermes‑style) — swap your own by matching the schema used in the notebooks.

**Tip:** Keep raw data in `data/` (ignored by Git) and log training runs to `wandb` or a simple CSV.

---

## Hardware Notes

* **12–16 GB VRAM**: BERT fine‑tuning, small LoRA runs (4‑bit).
* **24 GB VRAM**: Comfortable LoRA on 7B models.
* **48 GB+ VRAM / A100**: Full‑precision or larger batch sizes.
* Mixed precision (`bf16` on Ampere+) saves memory and usually speeds things up.

---

## Common Pitfalls

* **Model access denied**: Request on HF *and* login in the same environment; restart the kernel.
* **CUDA OOM**: Use `load_in_8bit/4bit`, reduce `batch_size`, enable gradient accumulation, shorten sequence length.
* **Tokenizer mismatch**: Always `AutoTokenizer.from_pretrained(same_model_id)` and save with `save_pretrained`.
* **Training too slow**: Turn on `torch.set_float32_matmul_precision("high")`, use `gradient_checkpointing`.

---

## Roadmap

* [ ] Add Mistral/Qwen SFT notebook
* [ ] Add QLoRA variant with paged optimizers
* [ ] Add evaluation harness (perplexity, exact‑match, Rouge‑L)
* [ ] Add small inference API (FastAPI + `transformers`/`vLLM`)

---

## Contributing

PRs welcome — especially improvements to clarity, comments, and reproducibility. If something breaks, open an issue with your **GPU model**, **driver/CUDA**, **PyTorch version**, and a **minimal repro**. Logs > guesses.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
