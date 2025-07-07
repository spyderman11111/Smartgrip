# Setting Up the Environment

```bash
conda activate smartgrip
```

---

# Git Submodule Setup


```bash
git submodule add <REPO_URL>

git submodule update --init --recursive
```
---

# Installation Instructions

## Install Grounded-SAM-2

```bash
cd Grounded-SAM-2
```

### Download checkpoints

```bash
cd checkpoints
bash download_ckpts.sh

cd gdino_checkpoints
bash download_ckpts.sh
```

### Install

```bash
pip install -e .
pip install --no-build-isolation -e grounding_dino
```
