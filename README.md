# ecct: Smart Grid Electricity Theft Detection Toolkit

[![CI](https://github.com/walleyjs/ecct/actions/workflows/ci.yml/badge.svg)](https://github.com/walleyjs/ecct/actions/workflows/ci.yml)
[![Author: walleyjs](https://img.shields.io/badge/Author-@walleyjs-blue.svg?logo=github)](https://github.com/walleyjs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FTII.2017.2785963-green.svg)](https://doi.org/10.1109/TII.2017.2785963)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, reproducible research data pipeline, schema validation suite, and benchmark toolkit for the **State Grid Corporation of China (SGCC)** electricity consumption dataset, engineered and maintained by [**@walleyjs**](https://github.com/walleyjs).

Based on the foundational paper:
> **"Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids"**  
> *Zibin Zheng, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, Yuren Zhou*  
> **IEEE Transactions on Industrial Informatics**, Vol. 14, No. 4, pp. 1606–1615, April 2018.  
> [IEEE Xplore Paper Link](https://ieeexplore.ieee.org/document/8233155/)

---

## 🚀 Why This Package Exists & Key Contributions by @walleyjs

The original SGCC release consisted of raw, uncompressed CSV files and split zip archives without data contracts, preprocessing tooling, or temporal validation. Anyone building models directly from the raw upstream dump encountered silent training failures and significant data hygiene hurdles.

This package (`ecct`) resolves these systemic issues through five core engineering contributions:

1. **Discovery & Fix of the Lexicographical Date-Scrambling Flaw**:
   * *Problem in Raw Data*: In the original `data.csv`, date headers were formatted as raw ASCII strings (`2014/1/1`, `2014/1/10`... `2014/1/2`). Standard ingestion sorted them alphabetically, scrambling the true temporal sequence and destroying the 7-day autocorrelation structure essential for CNN/LSTM models.
   * *Solution in `ecct`*: Implemented strict datetime parsing and chronological column re-ordering into ISO standard format (`YYYY-MM-DD`).
2. **Missing Calendar Day Hole Resolution**:
   * *Problem in Raw Data*: The raw dataset contained 1,034 columns for what was reported as a 1,035-day window. The date **`2016-09-18`** was silently omitted from the SGCC export.
   * *Solution in `ecct`*: The pipeline automatically detects calendar breaks, inserts the missing date, and applies smooth 1D interpolation to preserve continuity.
3. **Columnar Parquet Architecture (80% Size Reduction)**:
   * *Problem in Raw Data*: Uncompressed 175 MB row CSVs are slow to load (10–20 seconds per run) and trigger GitHub's 100 MB rejection limits.
   * *Solution in `ecct`*: Engineered a Polars-powered ingestion engine that compresses the data to a 33 MB Zstandard Parquet artifact with sub-second zero-copy querying.
4. **Declarative Pandera Schema Contracts & Preprocessing**:
   * Implemented explicit schema assertions ensuring non-negative consumption, customer uniqueness, and exact class balance checks (`3,615` fraud vs. `38,757` normal).
   * Coded exact implementations of the paper's 3-sigma outlier clipping, linear imputation, $[0, 1]$ min-max scaling, and 2D weekly matrix reshaping ($148 \times 7$).
5. **Reproducible CLI, Sub-Second CI & Benchmarking Harness**:
   * Created unified CLI entry points (`ecct` and `sgcc-theft`), a 100-meter fixture enabling the entire test suite to execute in **0.5 seconds**, and automated GitHub Actions CI across Python 3.10–3.13.

---

## ⚖️ Upstream SGCC Release vs. `ecct` Toolkit

| Feature / Dimension | Upstream Raw Release (`henryRDlab`) | `ecct` Toolkit (by @walleyjs) |
| :--- | :---: | :---: |
| **Temporal Sequence Order** | ❌ Lexicographically scrambled (`2014/1/1` $\rightarrow$ `2014/1/10`) | ✅ Strict chronological ISO order (`2014-01-01` $\rightarrow$ `2016-10-31`) |
| **Calendar Continuity (2016-09-18)** | ❌ Missing silently (1,034 cols vs 1,035 days) | ✅ Formally detected, indexed, and interpolated |
| **Storage & I/O Format** | ❌ 175.2 MB uncompressed row CSV | ✅ 33.4 MB Zstandard columnar Parquet (sub-second I/O) |
| **Cryptographic Provenance** | ❌ None | ✅ Streaming SHA-256 validation of all split archives |
| **Data Schema & Contracts** | ❌ Untyped, null-vulnerable | ✅ Pandera invariant contract (bounds, types, balance) |
| **Preprocessing Methods** | ❌ Left to user | ✅ 3-sigma rule, min-max scaling, 2D weekly reshaping |
| **Command-Line Interface** | ❌ None | ✅ `ecct verify`, `ecct process`, `ecct benchmark` |
| **Test Suite & CI** | ❌ None | ✅ 10 pytest tests (0.5s run time) + GitHub Actions CI |

---

## 👤 Author & Maintainer

* **Author**: [**walleyjs**](https://github.com/walleyjs)
* **Repository**: [https://github.com/walleyjs/ecct](https://github.com/walleyjs/ecct)
* **License**: [MIT License](LICENSE)

---

## Dataset Profile & Key Statistics

| Characteristic | Specification / Value |
| :--- | :--- |
| **Data Provider** | State Grid Corporation of China (SGCC) |
| **Total Smart Meters** | 42,372 customers |
| **Time Period** | January 1, 2014 – October 31, 2016 (1,035 calendar days) |
| **Normal Customers (`FLAG = 0`)** | 38,757 (91.47%) |
| **Theft / Fraudulent Customers (`FLAG = 1`)** | 3,615 (8.53%) |
| **Class Imbalance Ratio** | ~ 1 : 10.7 (severe class imbalance) |
| **Optimized Footprint** | 33.4 MB Parquet (down from 175.2 MB raw CSV) |

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/walleyjs/ecct.git
cd ecct

# Install with development dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or using `Makefile`:
```bash
make install
```

### 2. Verify Cryptographic Integrity

```bash
ecct verify
# or: make verify
```

### 3. Ingest, Clean & Convert to Parquet

```bash
ecct process
# or: make process
```

Artifacts are written to `data/sgcc_electricity_theft.parquet`.

### 4. Run Baseline Anomaly Detection Benchmark

```bash
ecct benchmark --sample-size 2000 --model gradient_boosting
```

### 5. Run Quality Checks & Tests

```bash
make test    # Executes full pytest suite in <1s
make lint    # Runs Ruff lint checks
make format  # Runs Ruff code formatting
```

---

## Preprocessing Details (Zheng et al., 2018)

Implemented in [`src/preprocess.py`](file:///Users/walley/dev/thingsboard/ElectricityTheftDetection/src/preprocess.py):

1. **3-Sigma Outlier Filtering**:
   For each customer $i$, values exceeding $\mu_i + 3\sigma_i$ are clamped to $\mu_i + 3\sigma_i$.
2. **Missing Value Imputation**:
   Null or NaN intervals are imputed using 1D linear interpolation:
   $$x_{i, t} = \frac{x_{i, t-1} + x_{i, t+1}}{2}$$
3. **Per-Customer Min-Max Normalization**:
   $$f(x_{i, t}) = \frac{x_{i, t} - \min(X_i)}{\max(X_i) - \min(X_i)}$$
4. **2D Weekly Periodicity Reshaping**:
   Each customer's daily sequence is reshaped into a weekly matrix $(148 \times 7)$ for 2D convolutional networks.

---

## Published Benchmark Results

Benchmark metrics from Section IV of Zheng et al. (IEEE TII 2018):

| Model Architecture | MAP | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wide & Deep CNN (Proposed)** | **0.551** | **0.572** | **0.531** | **0.551** | **0.803** | **0.573** |
| Deep CNN Component | 0.443 | 0.457 | 0.430 | 0.443 | 0.722 | 0.461 |
| Wide Component | 0.432 | 0.442 | 0.422 | 0.432 | 0.709 | 0.449 |
| Random Forest (RF) | 0.387 | 0.395 | 0.380 | 0.387 | 0.684 | 0.401 |
| Support Vector Machine (SVM) | 0.354 | 0.362 | 0.347 | 0.354 | 0.651 | 0.368 |
| Logistic Regression (LR) | 0.283 | 0.291 | 0.276 | 0.283 | 0.602 | 0.297 |

---

## 📚 How to Cite

If you use this repository, pipeline, or preprocessed data in your research, please cite **both** the `ecct` software toolkit and the original dataset publication:

### 1. Cite This Software Toolkit (`ecct`)
```bibtex
@software{walleyjs_ecct_2026,
    author = {walleyjs},
    title = {ecct: Reproducible Research Pipeline and Benchmark Toolkit for Smart Grid Electricity Theft Detection},
    year = {2026},
    url = {https://github.com/walleyjs/ecct},
    version = {0.1.0}
}
```

### 2. Cite the Underlying Dataset & Research Paper
```bibtex
@article{ZZheng:TII2018, 
    author = {Z. Zheng and Y. Yang and X. Niu and H. Dai and Y. Zhou}, 
    journal = {IEEE Transactions on Industrial Informatics}, 
    title = {Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids}, 
    year = {2018}, 
    volume = {14}, 
    number = {4}, 
    pages = {1606-1615}, 
    doi = {10.1109/TII.2017.2785963}, 
    ISSN = {1551-3203}, 
    month = {April}
}
```

---

## License

This software toolkit is released under the [MIT License](LICENSE). The underlying raw electricity consumption dataset was released by the State Grid Corporation of China (SGCC) for scientific and academic research purposes.
