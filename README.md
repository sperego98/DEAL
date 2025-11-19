# **DEAL**

**Data Efficient Active Learning for Machine Learning Potentials**

DEAL selects non-redundant structures from atomistic trajectories via Sparse Gaussian Processes. These structures can then be used to train machine-learning interatomic potentials (and more broadly, to identify informative configurations along reactive pathways).

The method is described in:

> **Perego S. & Bonati L.**
> *Data efficient machine learning potentials for modeling catalytic reactivity via active learning and enhanced sampling*,
> **npj Computational Materials 10, 291 (2024)**
> doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)

---

## Table of contents

- [🔧 Dependencies](#-dependencies)
- [🚀 Installation](#-installation)
- [🧪 Usage](#-usage)
  - [Minimal example](#minimal-example)
  - [With a YAML config file](#with-a-yaml-config-file)
  - [Python usage](#python-usage)
  - [Output files](#output-files)
  - [Multiple thresholds](#multiple-thresholds)

---

## 🔧 Dependencies

DEAL requires:

* [`flare`](https://github.com/mir-group/flare)
* `ase`
* `chemiscope`
* `pandas`
* `numpy`

---

## 🚀 Installation

Below is a complete installation sequence.

Create environment

```bash
conda create -n deal python=3.12
conda activate deal
```

Install **FLARE 1.3.3b**

(See: [https://mir-group.github.io/flare/installation/install.html](https://mir-group.github.io/flare/installation/install.html))

```bash
conda install -y gcc gxx cmake openmp liblapacke openblas -c conda-forge
git clone https://github.com/mir-group/flare.git -b 1.3.3b
cd flare
pip install .
cd ..
```

3. Install DEAL

```bash
git clone https://github.com/luigibonati/DEAL.git
cd DEAL
pip install .
```


---

## 🧪 Usage

DEAL can be run either with a command-line tool (`deal`) or using the python class (`DEAL`).

---


### ⭐ Minimal example

```bash
deal --file traj.xyz --threshold 0.1
```

DEAL will automatically:

* detect atomic species from the first frame
* use default GP/kernel/descriptor parameters
* use default output names


### 📄 With a YAML config file

For more customization you can create an `input.yaml` file:

```bash
deal -c input.yaml
```

```yaml
data:
  files: ["traj.xyz"]
  format: "extxyz"
  index: ":"

deal:
  threshold: 0.15
  output_prefix: "deal"

flare_calc:
  gp: SGP_Wrapper
  kernels:
    - name: NormalizedDotProduct
      sigma: 2
      power: 2
  descriptors:
    - name: B2
      nmax: 8
      lmax: 3
      cutoff_function: cosine
      radial_basis: chebyshev
  cutoff: 4.5
```

### 🐍 Python Usage

```python
# Import 
from deal import DataConfig, DEALConfig, FlareConfig, DEAL

# Define Config (uses defaults where not provided)
data_cfg = DataConfig(files="traj.xyz")
deal_cfg = DEALConfig(
    threshold=0.15,
    output_prefix="deal",    
)
flare_cfg = FlareConfig()

# Instantiate DEAL class
deal = DEAL(data_cfg, deal_cfg, flare_cfg)

# Run 
deal.run()

```

### Output files

In both cases the following files (with the default `output_prefix=deal`):

1. **`deal_selected.xyz` – selected frames** 

Contains the atomic configurations where the GP uncertainty exceeded the threshold.
Includes atoms.info["step"] indicating the original trajectory index.

2. **`deal_chemiscope.json.gz` – chemiscope visualization file**

Can be viewed online at https://chemiscope.org/ or inside Python:
```python
import chemiscope
chemiscope.show_input('deal_chemiscope.json.gz')
```

### Multiple thresholds

If the CLI receives a list of thresholds, DEAL will run once per threshold.
Equivalent behaviour in Python:

```python
for thr in [0.10, 0.15, 0.20]:
    deal_cfg.threshold = thr
    deal_cfg.output_prefix = f"run_thr{thr}"
    DEAL(data_cfg, deal_cfg, flare_cfg).run()
```



