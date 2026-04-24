# 🧭 Gaussian Process for Stage Inference (GPSI)

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-smoke%20passing-brightgreen.svg)](tests/test_smoke.py)
[![Method: Unsupervised](https://img.shields.io/badge/method-unsupervised-purple.svg)](gpsi/core.py)

[Quick Start](#-quick-start) • [Why GPSI](#-why-gpsi) • [What You Get](#-what-you-get) • [Method](#-method-overview) • [Use Your Data](#-using-your-own-data)

</div>

---


---

## 🚨 Why GPSI

Chronic disease progression is rarely observed as a clean timeline. In real
clinical datasets, patients are usually:

- 🧩 **Unordered**: visits and measurements do not come with a known disease stage
- 🕰️ **Irregularly observed**: clinical snapshots arrive at different times for different people
- 🧬 **Heterogeneous**: multiple progression patterns may exist inside the same diagnosis
- 📉 **Non-linear**: biomarkers and EHR features may worsen, plateau, or recover
- ❓ **Partly missing**: clinical features are not measured uniformly

GPSI addresses this setting by learning a latent disease stage from unordered,
high-dimensional data, while first separating patients into more homogeneous
subgroups.

---

## 👥 Who This Is For

- 🔬 Researchers studying chronic disease progression
- 🏥 EHR and real-world data analysts
- 🧠 Computational medicine and biomedical informatics teams
- 📊 ML researchers who need unsupervised disease staging
- 🎓 Students learning Gaussian processes for clinical phenotyping

---

## 🎁 What You Get

### 🧠 A Core GPSI Implementation

The package implements the workflow from the GPSI paper's Figure 1:

| Component | File | What It Does |
|---|---|---|
| Data processing | [`gpsi/preprocessing.py`](gpsi/preprocessing.py) | Feature filtering, median imputation, Z-score normalization |
| Subgroup discovery | [`gpsi/core.py`](gpsi/core.py) | K-means clustering and elbow-based subgroup selection |
| Stage inference | [`gpsi/core.py`](gpsi/core.py) | Per-subgroup GP latent stage inference |
| MCMC fitting | [`gpsi/core.py`](gpsi/core.py) | Metropolis-Hastings proposals plus hyperparameter updates |
| Simulation | [`gpsi/simulation.py`](gpsi/simulation.py) | Synthetic non-linear progression data |

### 🧪 Simulated Data Included

The repository ships with a small synthetic dataset:

```text
data/simulated/simulated_gpsi_data.csv
```

It includes:

- `subject_id`
- `true_cluster` and `true_stage` for evaluation only
- noisy non-linear features with missingness

The model does **not** use the true labels during fitting.

### 🧰 Example Scripts

```text
examples/run_simulated.py          # complete end-to-end demo
examples/generate_simulated_data.py
examples/ehr_adapter_template.py   # safe adapter boundary for private EHR data
```

### ✅ Smoke Test and GitHub Actions

```text
tests/test_smoke.py
.github/workflows/tests.yml
```

The test generates synthetic data, selects subgroups, and runs a short
Metropolis-Hastings GPSI fit.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/gpsi.git
cd gpsi

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional SHAP support:

```bash
pip install -e ".[interpretability,dev]"
```

### Run the Simulated Example

```bash
python examples/run_simulated.py \
  --input-csv data/simulated/simulated_gpsi_data.csv \
  --out-dir outputs/simulated \
  --mh-iterations 120 \
  --mh-burn-in 40
```

Expected outputs:

```text
outputs/simulated/
  simulated_gpsi_assignments.csv
  k_selection_scores.csv
  k_selection.png
  run_metadata.json
```

The assignment file contains:

| Column | Meaning |
|---|---|
| `subject_id` | synthetic subject identifier |
| `cluster` | inferred GPSI subgroup |
| `gpsi_pseudo_stage` | continuous latent stage in `[0, 1]` |
| `gpsi_stage_bin` | discrete stage from `ceil(s_i × smax)` |

---

## 🔬 Method Overview

GPSI models each subgroup separately. For subject `i` and feature `j`:

```text
x_i ~ N(w(s_i), Sigma)
w_j ~ GP(0, K_j)
K_j(s, s') = exp(-lambda_j^2 (s - s')^2)
s_i ~ Uniform(0, 1)
```

The implementation follows this procedure:

```text
Unordered raw data
        ↓
Feature selection + imputation + Z-score normalization
        ↓
K-means subgroup discovery
        ↓
Elbow method chooses number of subgroups
        ↓
For each subgroup: fit GPSI with MH-MCMC
        ↓
Output subgroup c and latent stage s
```

### Figure 1C Fitting

The `GPSI-MH` estimator implements:

1. Initialize `alpha`, `beta`, `gamma`
2. Initialize latent stage `s_i`, proposal variance `sigma_s2`, kernel rate `lambda_j2`, and noise variance `sigma_0j2`
3. Calculate GP log-likelihood
4. Propose new latent stages with Metropolis-Hastings
5. Accept or reject proposals
6. Update GP hyperparameters by MAP optimization
7. Return posterior mean latent stage

---

## 🧬 Using Your Own Data

Prepare a de-identified patient-level feature matrix:

```text
subject_id,age,feature_1,feature_2,feature_3
S0001,72,0.2,1.4,0.0
S0002,68,0.5,,1.1
```

Then run GPSI directly:

```python
import pandas as pd
from gpsi.preprocessing import preprocess_features
from gpsi.core import select_kmeans_clusters, assign_gpsi_stages_by_cluster

df = pd.read_csv("my_patient_features.csv")
feature_cols = [c for c in df.columns if c != "subject_id"]

X, selected_features, prep_meta = preprocess_features(df, feature_cols)

k_result = select_kmeans_clusters(X, k_min=2, k_max=8)
stage_result = assign_gpsi_stages_by_cluster(X, k_result.labels)

df["cluster"] = k_result.labels
df["gpsi_pseudo_stage"] = stage_result.pseudo_stage
df["gpsi_stage_bin"] = stage_result.stage_bin
```

---

## 📦 Repository Structure

```text
gpsi/
  core.py                  # GPSI model, elbow k selection, MH-MCMC
  preprocessing.py         # Figure 1A preprocessing
  simulation.py            # synthetic data generator
examples/
  run_simulated.py         # end-to-end simulated example
  generate_simulated_data.py
  ehr_adapter_template.py
data/simulated/
  simulated_gpsi_data.csv
tests/
  test_smoke.py
```

---

## 🧪 Tests

```bash
pytest -q
```

Expected:

```text
1 passed
```

---

## ⚠️ Notes

- This is an implementation of the GPSI workflow.
- The core model is unsupervised. Truth labels in the simulated dataset are
  included only for evaluation.

---

## 🌟 Citation

If this repository helps your research, please cite the following paper:

Wang Y, Zhao W, Ross A, You L, Wang H, Zhou X. **Revealing chronic disease
progression patterns using Gaussian process for stage inference.** *Journal of
the American Medical Informatics Association*. 2024;31(2):396-405.

```bibtex
@article{wang2024gpsi,
  title   = {Revealing chronic disease progression patterns using Gaussian process for stage inference},
  author  = {Wang, Yanfei and Zhao, Weiling and Ross, Angela and You, Lei and Wang, Hongyu and Zhou, Xiaobo},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {31},
  number  = {2},
  pages   = {396--405},
  year    = {2024}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if GPSI helps you uncover hidden disease progression patterns. ⭐**

</div>
