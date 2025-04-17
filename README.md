![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
# MMFF94 in PyTorch

This repository provides a PyTorch implementation of the MMFF94 force field. 
It allows molecular energy minimization both in isolation and conditioned on a protein pocket.

## Features

- PyTorch-based implementation of MMFF94
- Molecular energy minimization
- Optional conditioning on protein pocket environments
- Simple interface for testing and running examples

---

## 🚀 Installation

Clone the repository and install the dependencies:

```bash
$ pip install -r requirements.txt
```

---

## 📦 Running Examples

You can run an example script with:

```bash
$ python run_examples.py
```

This will demonstrate basic usage, including standalone molecule minimization and 
pocket-conditioned optimization.

---

## ✅ Testing

To run the test suite:

```bash
pytest tests
```

This will execute all unit tests to ensure correct behavior.