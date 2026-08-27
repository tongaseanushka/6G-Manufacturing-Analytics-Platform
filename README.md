# Thales 6G Manufacturing Analytics Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://6g-manufacturing-analytics-platform-5hcf4d6vsmzskvftavb9nd.streamlit.app/)

> A full-stack data analytics platform that investigates the statistical relationship between **6G network performance** and **smart manufacturing efficiency** using 100,000 rows of industrial telemetry data.

**🔴 Live Dashboard:** [Click here to view the deployed application](https://6g-manufacturing-analytics-platform-5hcf4d6vsmzskvftavb9nd.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [KPIs Computed](#kpis-computed)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Reports](#reports)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

This platform analyzes telemetry from **50 industrial machines over ~69 days** to answer:

> *Is there a statistically meaningful relationship between 6G network conditions (latency, packet loss) and manufacturing efficiency outcomes (efficiency class, error rates, defect rates)?*

Four custom Key Performance Indicators (KPIs) are computed using rigorous aggregated statistical methods — including chi-square tests, Welch's t-test, LOWESS smoothing, and Spearman correlation — and results are surfaced through:

- An **interactive Streamlit dashboard** with real-time filters
- A **full technical research paper** (Markdown + embedded charts)
- A **plain-language executive summary** for non-technical stakeholders

---

## Features

| Feature | Details |
|---|---|
| Interactive Dashboard | 4 analytical modules, sidebar filters, Plotly charts |
| Shared KPI Engine | Single source of truth — same numbers in dashboard and reports |
| Research Paper Generator | CLI tool → full Markdown paper with LaTeX formulas and charts |
| Executive Summary Generator | ≤800-word, jargon-free summary for government/business stakeholders |
| Statistical Rigor | Chi-square, Cramer's V, Welch's t-test, LOWESS, Spearman correlation |
| 64 Tests Passing | Unit tests + property-based tests via Hypothesis |
| Edge Case Handling | NaN propagation, zero-denominator guards, degenerate input handling |

---

## Architecture

```
CSV Dataset
    |
    v
data_prep.py            <- Load, validate, classify Network_Quality_Band
    |
    |---> kpi_computation.py      <- NSI, LSS, PLIR, NEC  (single source of truth)
    |           |
    |           |---> dashboard.py                   -> http://localhost:8501
    |           `---> paper_generator.py             -> reports/research_paper.md
    |                      |
    |                      `---> executive_summary_generator.py -> reports/executive_summary.md
    |
    `---> statistical_analysis.py   <- Spearman, LOWESS, Pearson, class distribution
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas >= 2.0, scipy, statsmodels |
| Dashboard | Streamlit >= 1.28 |
| Visualization | Plotly, matplotlib, seaborn, altair |
| Testing | pytest, hypothesis (property-based) |

---

## Project Structure

```
project/
|
|-- src/
|   |-- data_prep.py                    # Data loading, validation, Network_Quality_Band
|   |-- kpi_computation.py              # NSI . LSS . PLIR . NEC - shared KPI engine
|   |-- statistical_analysis.py         # Spearman, LOWESS, Pearson, class distribution
|   |-- dashboard.py                    # Streamlit interactive dashboard (main app)
|   |-- paper_generator.py              # Technical research paper generator
|   |-- executive_summary_generator.py  # Plain-language executive summary generator
|   `-- report_gen.py                   # CLI entry point for report generation
|
|-- tests/
|   |-- test_data_prep.py               # 10 validation unit tests
|   |-- test_data_prep_basic.py         # 10 network quality classification tests
|   |-- test_data_prep_performance.py   # Loading performance test (< 5s)
|   `-- test_kpi_and_stats.py           # 43 KPI + stat tests (incl. Hypothesis)
|
|-- data/
|   |-- README.md                       # Dataset column descriptions
|   `-- Thales_Group_Manufacturing.csv  # NOT tracked in git - add your own
|
|-- reports/
|   |-- research_paper.md               # Generated full technical paper
|   |-- executive_summary.md            # Generated executive summary
|   `-- figures/                        # Auto-generated PNG charts
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Dataset

Place `Thales_Group_Manufacturing.csv` in the `data/` folder.

| Property | Value |
|---|---|
| Rows | 100,000 |
| Columns | 14 raw + 1 derived (Network_Quality_Band) |
| Machines | 50 unique IDs |
| Time Range | ~69 days (2025-01-01 to 2025-03-10) |
| Date Format | DD-MM-YYYY |

**Required Columns:**

| Column | Type | Description |
|---|---|---|
| Timestamp | Date | DD-MM-YYYY format |
| Machine_ID | String | Unique machine identifier |
| Network_Latency_ms | Float | Network latency in ms (>= 0) |
| Packet_Loss_% | Float | Packet loss percentage (0-100) |
| Efficiency_Status | Categorical | Low / Medium / High |
| Production_Speed | Float | Production rate |
| Error_Rate | Float | Manufacturing error rate |
| Quality_Control_Defect_Rate | Float | Quality defect rate |
| Operation_Mode | Categorical | Active / Idle / Maintenance |

**Network Quality Band Classification (6G URLLC thresholds):**

| Band | Latency | Packet Loss |
|---|---|---|
| High | < 20 ms AND | < 1% |
| Medium | < 50 ms AND | < 5% |
| Low | >= 50 ms OR | >= 5% |

---

## KPIs Computed

All KPIs use **aggregated (binned/grouped) analysis** because row-level correlations are near-zero in this dataset. The `kpi_computation.py` module is the single source of truth used by both the dashboard and report generators.

### 1. Network Stability Index (NSI)

Measures how consistent network performance is over time.

```
NSI = 1 - (CV_latency + CV_packet_loss) / 2
CV  = min(std / mean, 1.0)          <- coefficient of variation, capped at 1
Range: 0 (unstable) -> 1 (perfectly stable)
```

### 2. Latency Sensitivity Score (LSS)

Quantifies efficiency change with latency using linear regression on 10 decile bins.

```
Encode: Low=1, Medium=2, High=3
Bin latency into 10 deciles -> mean efficiency per bin
LSS = |slope| of mean_efficiency ~ mean_latency_per_bin
```

### 3. Packet Loss Impact Ratio (PLIR)

Compares error rates between high (Q4) and low (Q1) packet-loss groups via Welch's t-test.

```
PLIR = mean_error_rate(Q4 packet loss) / mean_error_rate(Q1 packet loss)
```

### 4. Network-Efficiency Correlation (NEC)

Tests association between Network_Quality_Band and Efficiency_Status using chi-square + Cramer's V.

```
V = sqrt(chi2 / (n * (min(rows, cols) - 1)))     <- Cramer's V [0, 1]
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/thales-6g-analytics.git
cd thales-6g-analytics

# 2. Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your dataset
# Copy Thales_Group_Manufacturing.csv into the data/ folder
```

---

## Usage

### Run the Interactive Dashboard

```bash
python -m streamlit run src/dashboard.py
```

Opens at **http://localhost:8501**

**Dashboard modules:**
1. **Network Performance Overview** - Daily latency/packet-loss trends + KPI scorecards
2. **Network vs Efficiency** - Stacked bar chart + scatter with OLS trendline
3. **Quality and Error Impact** - Error rate analysis by network quality band
4. **Optimization Insights** - Risk heatmap + threshold-based recommendations

**Sidebar filters:** Network Quality Band | Efficiency Class | Operation Mode | Date Range

### Generate Research Paper

```bash
python src/report_gen.py --output reports/research_paper.md
```

### Generate Executive Summary

```bash
python src/report_gen.py --summary --output reports/executive_summary.md
```

### Custom Dataset Path

```bash
python src/report_gen.py --csv path/to/data.csv --output reports/paper.md
```

---

## Running Tests

```bash
# Run all 64 tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_kpi_and_stats.py -v

# Run only fast unit tests (skip Hypothesis property tests)
python -m pytest tests/ -v -k "not always_bounded"
```

**Test Results: 64 / 64 PASSED**

| Test File | Tests | What It Covers |
|---|---|---|
| test_data_prep.py | 10 | Schema validation, exceptions, date parsing |
| test_data_prep_basic.py | 10 | Network quality classification and boundary cases |
| test_data_prep_performance.py | 1 | Dataset loads in < 5 seconds |
| test_kpi_and_stats.py | 43 | All 4 KPIs + stat functions (unit + Hypothesis property tests) |

---

## Reports

Auto-generated by the CLI tool into `reports/`:

| File | Description |
|---|---|
| reports/research_paper.md | Full technical paper - abstract, methodology, LaTeX KPI formulas, results, figures, discussion, limitations, references |
| reports/executive_summary.md | <=800-word plain-language summary - key findings, 5 recommendations, synthetic data caveats |
| reports/figures/ | PNG charts: efficiency distribution, latency histogram, risk heatmap, executive bar chart |

---

## Deployment

### Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and click **New App**
3. Select your repo and set **Main file path** to `src/dashboard.py`
4. Click **Deploy**

> **Dataset note:** `data/*.csv` is excluded from git via `.gitignore`. The dashboard includes a built-in **file uploader** fallback - if no CSV is found on disk it prompts you to upload one directly in the browser.

---

## License

MIT License - free to use and adapt for research and commercial purposes.

---

*Thales 6G Manufacturing Analytics Platform - Built with Python and Streamlit*
