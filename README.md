Domain Adaptation In Galaxy Morphology and Star Formation Rates
======================================

Domain Adaptation for Galaxy Morphology Classification using llustrisTNG and Galaxy Zoo Evolution dataset

This work was presented at NeurIPS 2025 Machine Learning and the Physical Sciences workshop.
The paper can be found: https://arxiv.org/abs/2511.18590 

## Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Installation

1. Clone this [`repo`](https://github.com/kaleybrauer/galaxy_da.git)

```bash
git clone https://github.com/kaleybrauer/galaxy_da.git
```

2. Install dependencies:

```bash
make install
```

- or you can create a `.venv`:

```sh
python3 -m venv .venv
source .venv/bin/activate # On mac/linux distros
```
- Install `nebula`

```sh
pip install -e .
```

## How to train?

1. Create a config file, see [template](./configs/config.template.yml) and run with 

```sh
python3 scripts/run_train.py --config /path/to/config.yml
```

## How to evaluate?

```sh
python3 scripts/run_eval.py /path/to/ckpt
```
You can also run train and evaluate simultaneously. Run this with a single config, multiple configs, or a folder of configs by passing `-f`:

```sh
./run_experiment.sh <config_path> [more_configs...]
# or for a folder of configs:
./run_experiment.sh -f <config_folder>
```
## About This Project

This project was made possible through the [2025 IAIFI Summer School](https://github.com/iaifi/summer-school-2025) provided by The [NSF AI](https://iaifi.org/) Institute for Artificial Intelligence and Fundamental Interactions (IAIFI).

### Team Members

- [@ahmedsalim3](https://ahmedsalim3.github.io/)
- [@Meet-Vyas-Dev](https://meet-vyas-dev.github.io/)
- [@kaleybrauer](https://www.kaleybrauer.com/)
- [@adityadash54](https://github.com/adityadash54)
- [@stivenbg](https://github.com/stivenbg)
- [@dingq1](https://github.com/dingq1)
- [@AumRTrivedi](https://github.com/AumRTrivedi)

### References
