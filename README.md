# Expectation Maximisers

Course project combining **noisy crowd preferences** on [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) with **Dawid–Skene-style expectation maximization (EM)** and **EM-weighted DPO-style** preference optimization on a language model.

Annotators vote on which of two responses (A or B) is better; votes are simulated with heterogeneous reliability. EM infers latent “which side is truly better” per prompt and worker parameters. A **policy** model (GPT-2 or a small dummy LM) supplies **LLM priors** for the E-step; training uses a **weighted** objective where soft labels come from EM’s posterior over A vs B.

---

## Quick results (from prior runs)

- **~75% golden-dataset accuracy** when using the real **GPT-2** policy path (`train_joint.py`): the model’s prior (sigmoid of implicit reward vs a frozen reference) is compared to the simulator’s hidden `truth_is_A` each epoch.
- **`train_joint_wo_sft.py`** uses a **stub causal LM** (no pretrained SFT weights): faster to run, weaker language signal, and **lower** golden accuracy than the GPT-2 pipeline.

Exact numbers depend on simulation seed, hardware, and hyperparameters.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

Dependencies: PyTorch, Transformers, Datasets, Pandas, NumPy, tqdm (see `requirements.txt`).

**GPU:** Training picks CUDA when available, else Apple MPS, else CPU (`src/utils/device.py`). Pre-tokenization is CPU-heavy; use GPU for `train_joint.py` if you can ([PyTorch CUDA builds](https://pytorch.org/get-started/locally/)).

---

## How to run (important: working directory)

Paths like `../../data/...` are resolved relative to your **shell’s current directory**, not relative to the script file. Run **data** scripts from `src/data` and **training** scripts from `src/training` so outputs land under the repo’s `data/` folder.

### 1. Simulate votes and ground truth

```bash
cd src/data
python simulator.py
```

**Outputs** (under `data/` at the repo root):

| Path | Content |
|------|--------|
| `data/processed/simulated_noisy_votes.csv` | Per (prompt, annotator) binary vote (1 = A, 0 = B) |
| `data/processed/ground_truth.csv` | Hidden `truth_is_A` per prompt (evaluation / “golden” labels) |
| `data/true_params/true_annotator_parameters.csv` | Simulated per-worker accuracy (for analysis; EM does not read this during training) |

Requires downloading the HH-RLHF dataset slice from Hugging Face.

### 2. Pre-tokenize for the GPT-2 pipeline (optional but recommended for `train_joint.py`)

```bash
cd src/data
python prepare_tensors.py
```

**Output:** `data/tokenized/rlhf_tokens.pt` — fast loading for `PreTokenizedRlhfDataset`.

Skip this step if you only run `train_joint_wo_sft.py` (it tokenizes on the fly).

### 3. Joint EM + weighted preference training

**Full model (GPT-2 policy + frozen GPT-2 reference):**

```bash
cd src/training
python train_joint.py
```

The terminal prints **golden dataset accuracy** (LLM prior vs `truth_is_A`) and **weighted DPO loss** (and reward margin) per epoch.

**Ablated / fast path (dummy LM, on-the-fly tokenization):**

```bash
cd src/training
python train_joint_wo_sft.py
```

Same alternating structure (LLM priors → EM → weighted loss) but a small randomly initialized transformer instead of pretrained GPT-2.

### Other utilities

| Command | Purpose |
|---------|--------|
| `cd src/training && python em_only.py` | EM on votes **only** (no LLM); compares inferred labels and worker stats to CSV ground truth / true params |
| `cd src/training && python test_dpo.py` | Smoke test: forward/backward through dummy LMs + `weighted_dpo_loss` |

---

## Project layout

```text
data/                          # created when you run simulators (see .gitignore)
  processed/
  tokenized/
  true_params/
notebooks/
  testing/                     # exploratory analysis and plots (e.g. baseline_demo_01.ipynb)
src/
  data/
    simulator.py               # bipartite assignment + noisy votes on HH-RLHF
    prepare_tensors.py         # GPT-2 tokenization → rlhf_tokens.pt
    dataset.py                 # Pre-tokenized PyTorch Dataset
    dataset_wo_sft.py          # On-the-fly tokenized Dataset
  models/
    em.py                      # DawidSkeneEM (binary votes, optional LLM priors in E-step)
    dpo.py                     # sequence log-probs + weighted DPO loss
    sft_model.py               # load GPT-2 (policy / reference)
    dummy_llm.py               # small causal LM for ablations
  training/
    train_joint.py             # GPT-2 joint training + metrics history dict
    train_joint_wo_sft.py      # dummy LM joint training
    em_only.py                 # EM baseline without LLM
    test_dpo.py                # loss pipeline sanity check
  utils/
    device.py                  # cuda / mps / cpu
    metrics.py                 # golden accuracy, reward margin, worker α snapshots
requirements.txt
README.md
```

---

## Method sketch

1. **Simulation:** Random (l,r)-regular-style bipartite graph between tasks and annotators; each worker has a latent accuracy; votes follow that accuracy against the hidden true A/B assignment.
2. **E-step:** Posterior γ over “A is truly better” per prompt, using votes and per-worker α, β; optionally **conditioned** on LLM-derived priors from the current policy vs reference.
3. **M-step:** Update π₁, α, β from expected counts (vectorized with `scatter_add`).
4. **Policy update:** Weighted DPO-style loss: γ weights the loss as if A were preferred, (1−γ) as if B were preferred.

---

## Notebooks

Use **`notebooks/testing/`** (and other notebook folders you add) for **plots, tables, and ad-hoc analysis** on the generated CSVs, `.pt` tensors, or saved training logs. The main training loop prints metrics to the terminal; persisting plots in the notebook keeps the repo narrative clear for coursework.

---

## Data in Git

Processed CSVs, tokenized `.pt` files, and similar artifacts are listed in `.gitignore`. After cloning, always run **step 1** (and **step 2** if using GPT-2 pre-tokenization) to regenerate local data.

---
