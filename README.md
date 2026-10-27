# SGCC Electricity Theft Detection Benchmark

[![CI](https://github.com/henryRDlab/ElectricityTheftDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/henryRDlab/ElectricityTheftDetection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FTII.2017.2785963-green.svg)](https://doi.org/10.1109/TII.2017.2785963)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible research data pipeline, schema validation suite, and benchmark toolkit for the **State Grid Corporation of China (SGCC)** electricity consumption dataset, as presented in:

> **"Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids"**  
> *Zibin Zheng, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, Yuren Zhou*  
> **IEEE Transactions on Industrial Informatics**, Vol. 14, No. 4, pp. 1606–1615, April 2018.  
> [IEEE Xplore Paper Link](https://ieeexplore.ieee.org/document/8233155/)

---

## Dataset Profile & Key Statistics

The SGCC dataset provides real-world daily smart-meter readings spanning nearly three years.

| Characteristic | Specification / Value |
| :--- | :--- |
| **Data Provider** | State Grid Corporation of China (SGCC) |
| **Total Smart Meters** | 42,372 customers |
| **Time Period** | January 1, 2014 – October 31, 2016 (1,035 calendar days) |
| **Normal Customers (`FLAG = 0`)** | 38,757 (91.47%) |
| **Theft / Fraudulent Customers (`FLAG = 1`)** | 3,615 (8.53%) |
| **Class Imbalance Ratio** | ~ 1 : 10.7 (severe class imbalance) |
| **Raw Storage Footprint** | 175.2 MB (CSV) $\rightarrow$ 33.4 MB (compressed Parquet) |

### Important Dataset Quirks & Fixes

1. **Non-Chronological Header Ordering in Raw CSV**:
   In the original `data.csv`, date headers are string-formatted (`2014/1/1`, `2014/1/10`, ..., `2014/1/2`) and dumped in lexicographical ASCII order rather than chronological date order. Naive sequential loading feeds temporally scrambled sequences to models. This pipeline parses and re-orders all time steps into strict chronological ISO dates (`YYYY-MM-DD`).
2. **Missing Calendar Date**:
   The raw dataset contains 1,034 date columns spanning a 1,035-day window. The date **`2016-09-18`** is missing from the SGCC export. This pipeline automatically aligns the calendar index and inserts the date to preserve time continuity.
3. **Multipart Zip Archives**:
   To comply with Git repository file limits, the original dataset is partitioned into `data.z01`, `data.z02`, and `data.zip`. The automated pipeline verifies cryptographic SHA-256 digests and reconstructs the uncompressed dataset deterministically.

---

## Repository Structure

```text
ElectricityTheftDetection/
├── .github/
│   └── workflows/
│       └── ci.yml               # Multi-version Python CI (lint, format, pytest)
├── .gitignore                   # Excludes raw 175MB CSVs, caches, and artifacts
├── Makefile                     # Standard developer commands (install, test, process)
├── pyproject.toml               # PEP 518/621 configuration, dependencies, and tools
├── README.md                    # Technical documentation and dataset card
├── data.z01                     # Multipart raw archive (part 1)
├── data.z02                     # Multipart raw archive (part 2)
├── data.zip                     # Multipart raw archive (part 3)
├── src/
│   ├── __init__.py
│   ├── baseline.py              # Stratified benchmark evaluation (PR-AUC, ROC-AUC, F1)
│   ├── cli.py                   # Command-line interface (`sgcc-theft`)
│   ├── config.py                # Dataset metadata, hashes, and constants
│   ├── ingestion.py             # Polars-backed ingestion, chronological sort, and Parquet export
│   ├── integrity.py             # SHA-256 checksums and split-zip concatenation
│   ├── preprocess.py            # 3-sigma clipping, imputation, min-max, and weekly 2D reshaping
│   └── schema.py                # Pandera data contract and invariant checks
└── tests/
    ├── conftest.py              # Test fixtures
    ├── fixtures/
    │   └── sample_100_meters.csv # Fast CI test fixture (100 meters)
    ├── test_ingestion.py
    ├── test_integrity.py        # Validates SHA-256 hashes against SGCC distribution
    ├── test_preprocess.py       # Validates 3-sigma, scaling, and 2D matrix transformation
    └── test_schema.py           # Validates schema contracts and date ordering
```

---

## Quickstart

### 1. Installation

Create a virtual environment and install the package with development dependencies:

```bash
# Using standard Python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Or using uv (recommended for speed)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or simply use the `Makefile`:
```bash
make install
```

### 2. Verify Cryptographic Integrity

Validate the SHA-256 digests of the multipart split archives before extraction:

```bash
make verify
# or
sgcc-theft verify
```

Expected output:
```text
Verifying cryptographic SHA-256 digests for raw multipart split volumes...
  - data.z01     : PASSED [OK]
  - data.z02     : PASSED [OK]
  - data.zip     : PASSED [OK]

[SUCCESS] All multipart archives verified successfully.
```

### 3. Ingest & Convert to Columnar Parquet

Reconstruct the archive, order dates chronologically, validate schema invariants, and compress:

```bash
make process
# or
sgcc-theft process
```

The output is written to `data/sgcc_electricity_theft.parquet`. Storage is reduced by ~80% with instantaneous zero-copy columnar scanning.

### 4. Run Tests & Linter

```bash
make test    # Runs pytest suite (<1 second via 100-meter fixture)
make lint    # Runs Ruff linter
make format  # Runs Ruff formatter
```

---

## Preprocessing Pipeline (Zheng et al., 2018)

The package provides exact implementations of the paper's data preparation methodology (`src/preprocess.py`):

1. **3-Sigma Outlier Filtering**:
   For each customer $i$, consumption values exceeding $\mu_i + 3\sigma_i$ are identified as meter malfunction outliers and clipped to $\mu_i + 3\sigma_i$.
2. **Missing Value Imputation**:
   Null or NaN intervals are imputed using 1D linear interpolation from neighboring valid time steps:
   $$x_{i, t} = \frac{x_{i, t-1} + x_{i, t+1}}{2}$$
3. **Per-Customer Min-Max Scaling**:
   Because electricity consumption varies across residential and commercial accounts, each customer's sequence is normalized independently to $[0, 1]$:
   $$f(x_{i, t}) = \frac{x_{i, t} - \min(X_i)}{\max(X_i) - \min(X_i)}$$
4. **2D Weekly Periodicity Reshaping**:
   To leverage 2D convolutional networks, each customer's 1D daily vector of length $T$ (1,035 days) is reshaped into a weekly periodic matrix of shape $(W \times 7)$ where $W = 148$ weeks.

---

## Published Benchmark Performance

Below are the benchmark metrics reported in Zheng et al. (IEEE TII 2018) on the SGCC dataset under imbalanced conditions:

| Model Architecture | MAP | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wide & Deep CNN (Proposed)** | **0.551** | **0.572** | **0.531** | **0.551** | **0.803** | **0.573** |
| Deep CNN Component | 0.443 | 0.457 | 0.430 | 0.443 | 0.722 | 0.461 |
| Wide Component | 0.432 | 0.442 | 0.422 | 0.432 | 0.709 | 0.449 |
| Random Forest (RF) | 0.387 | 0.395 | 0.380 | 0.387 | 0.684 | 0.401 |
| Support Vector Machine (SVM) | 0.354 | 0.362 | 0.347 | 0.354 | 0.651 | 0.368 |
| Logistic Regression (LR) | 0.283 | 0.291 | 0.276 | 0.283 | 0.602 | 0.297 |

### Run Baseline Model

To run a rapid stratified baseline evaluation on local data:

```bash
sgcc-theft benchmark --sample-size 2000 --model gradient_boosting
```

---

## Citation

If you use this dataset or code in your research, please cite the original publication:

### Plain Text
```text
Zibin Zheng, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, Yuren Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids", IEEE Transactions on Industrial Informatics, vol. 14, no. 4, pp. 1606-1615, April 2018.
```

### BibTeX
```bibtex
@ARTICLE{ZZheng:TII2018, 
    author={Z. Zheng and Y. Yang and X. Niu and H. Dai and Y. Zhou}, 
    journal={IEEE Transactions on Industrial Informatics}, 
    title={Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids}, 
    year={2018}, 
    volume={14}, 
    number={4}, 
    pages={1606-1615}, 
    doi={10.1109/TII.2017.2785963}, 
    ISSN={1551-3203}, 
    month={April}
}
```

---

## License

This repository and pipeline tooling are released under the [MIT License](LICENSE). The underlying dataset was released by the State Grid Corporation of China (SGCC) for scientific and academic research purposes.
