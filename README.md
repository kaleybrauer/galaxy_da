# Simulation→Observation Domain Adaptation for Galaxy Morphology


Train on **simulated** galaxies (TNG50), adapt to **real** DESI galaxies with MMD / Sinkhorn / DANN. 


---

## Data access & layout

**Data Access OSF link:** https://osf.io/mxhe6/?view_only=52e8633869984ed8a54cea0610ab91f5

Download into **`data/source`** and **`data/target`**

- `galaxy_images_rgb.zip` — **Source** (TNG50 mocks, RGB PNGs).  
- `gz_desi.zip` — **Target** (DESI galaxies, RGB PNGs).  

**Labels (CSV files):**

- **`data/source/galaxy_labels.csv`** — labels and metadata for **source** (TNG50) mock galaxies.
- **`data/target/gz_desi_labels.csv`** — labels and metadata for **target** (DESI) real galaxies.

---

## Methods

- **Architecture:** compact CNN → GAP → MLP; 128-D penultimate feature.  
  Source cross-entropy uses **class weighting** (ellipticals upweighted) to reflect target priors.
- **Alignment (unsupervised target):**  
  **MMD/energy** (`SamplesLoss("energy", p=2)`),  
  **Sinkhorn OT** (`SamplesLoss("sinkhorn", p=2, blur, debias=True)` with **blur=0.75**),  
  **DANN** (domain discriminator with **GRL**; GRL coeff **0.1**).  
  Features are **L2-normalized** before alignment losses.
- **Main hyperparams:** `--lambda 0.1` (MMD/Sinkhorn); `--adv 0.1` (DANN); **10 epochs**; batch size **32**.  
  Diagnostics include Gaussian **MMD²**, **Sinkhorn divergence**, and **domain-probe AUC**.

---

## Reproducing paper figures

Overview of the model can be found in **`model/model_overview.ipynb`**

To reproduce the paper figures, run:

```bash
python scripts/make_plots.py --metrics outputs/all_methods_10epochs.csv --out outputs/figs --small

```
