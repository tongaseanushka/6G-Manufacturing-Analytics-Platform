"""
Research Paper Generator for Thales 6G Manufacturing Analytics Platform.

This module generates a full technical research paper in Markdown format, using
the shared KPI computation module as the single source of truth for all reported
metrics. The paper follows standard academic structure and documents all formulas,
methods, and findings transparently, including null or weak results.

Usage:
    from paper_generator import generate_research_paper
    paper_md = generate_research_paper(df, kpis)

Requirements Addressed:
    - 15.1–15.9: Research paper generation
    - 17.7: Statistical methods documented
    - 20.3, 20.6: Docstrings and type hints
    - 22.1–22.5: Synthetic data transparency
    - 23.5: Single source of truth (KPI values from compute_all_kpis)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kpi_computation import AllKPIs
from statistical_analysis import compute_class_distribution

# Figures output directory
FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"


def _ensure_figures_dir() -> Path:
    """Create the figures output directory if it does not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _fmt(value: float, fmt: str = ".4f") -> str:
    """Format a float value; return 'N/A' for NaN."""
    if math.isnan(value):
        return "N/A"
    return format(value, fmt)


# ---------------------------------------------------------------------------
# Static figure generation
# ---------------------------------------------------------------------------


def _generate_efficiency_distribution_figure(df: pd.DataFrame, fig_dir: Path) -> str:
    """Generate and save efficiency distribution by network quality bar chart.

    Returns:
        Relative path string for embedding in Markdown.
    """
    fig_path = fig_dir / "efficiency_distribution.png"
    ct = pd.crosstab(df["Network_Quality_Band"], df["Efficiency_Status"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"}
    bottom = np.zeros(len(ct))

    for col in ["Low", "Medium", "High"]:
        if col in ct.columns:
            vals = ct[col].values
            ax.bar(ct.index, vals, bottom=bottom, label=col, color=colors.get(col, "#999"))
            bottom += vals

    ax.set_xlabel("Network Quality Band")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Efficiency Class Distribution by Network Quality Band")
    ax.legend(title="Efficiency Status")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path.relative_to(fig_path.parent.parent))


def _generate_latency_distribution_figure(df: pd.DataFrame, fig_dir: Path) -> str:
    """Generate and save network latency histogram.

    Returns:
        Relative path string for embedding in Markdown.
    """
    fig_path = fig_dir / "latency_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Network_Latency_ms"], bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Network Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Network Latency")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path.relative_to(fig_path.parent.parent))


def _generate_risk_heatmap_figure(df: pd.DataFrame, fig_dir: Path) -> str:
    """Generate and save efficiency risk heatmap.

    Returns:
        Relative path string for embedding in Markdown.
    """
    fig_path = fig_dir / "risk_heatmap.png"
    try:
        df_hm = df.copy()
        df_hm["lat_q"] = pd.qcut(df_hm["Network_Latency_ms"], q=5, duplicates="drop", labels=False)
        df_hm["pl_q"] = pd.qcut(df_hm["Packet_Loss_%"], q=5, duplicates="drop", labels=False)
        pivot = (
            df_hm.groupby(["lat_q", "pl_q"], observed=True)["Efficiency_Status"]
            .apply(lambda s: (s == "Low").mean() * 100)
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
        )
        ax.set_xlabel("Packet Loss Quintile (0=Low, 4=High)")
        ax.set_ylabel("Latency Quintile (0=Low, 4=High)")
        ax.set_title("Low Efficiency Rate (%) by Network Conditions")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")
    return str(fig_path.relative_to(fig_path.parent.parent))


# ---------------------------------------------------------------------------
# Paper sections
# ---------------------------------------------------------------------------


def _section_title_abstract(kpis: AllKPIs) -> str:
    """Generate title, authors placeholder, and abstract section."""
    nsi = _fmt(kpis.nsi.nsi, ".3f")
    cramers = _fmt(kpis.nec.cramers_v, ".3f")
    return f"""# Statistical Analysis of 6G Network Performance Impact on Smart Manufacturing Efficiency

**Date**: {datetime.now().strftime("%B %Y")}  
**Platform**: Thales 6G Manufacturing Analytics Platform  
**Dataset**: Thales_Group_Manufacturing.csv (100,000 rows, 50 machines, ~69 days)

---

## Abstract

This study investigates the statistical relationship between 6G network performance metrics
(latency and packet loss) and smart manufacturing efficiency outcomes (efficiency status,
error rates, quality defect rates) using a 100,000-row historical telemetry dataset from
50 industrial machines over approximately 69 days.

Because row-level correlations between network and manufacturing metrics are approximately
zero in this dataset, all analyses employ aggregated (binned/grouped) methods. Four Key
Performance Indicators are computed: the Network Stability Index (NSI={nsi}),
Latency Sensitivity Score (LSS), Packet Loss Impact Ratio (PLIR), and the
Network-Efficiency Correlation via chi-square test and Cramér's V (V={cramers}).

Findings are reported transparently regardless of statistical significance. The dataset
exhibits characteristics consistent with synthetic or simulated data (clean ranges,
no missing values), which limits the generalizability of findings to real-world deployments.

---
"""


def _section_introduction(df: pd.DataFrame) -> str:
    """Generate the introduction section."""
    n_rows = len(df)
    n_machines = df["Machine_ID"].nunique() if "Machine_ID" in df.columns else "N/A"
    date_min = df["Timestamp"].min().strftime("%Y-%m-%d") if "Timestamp" in df.columns else "N/A"
    date_max = df["Timestamp"].max().strftime("%Y-%m-%d") if "Timestamp" in df.columns else "N/A"
    return f"""## 1. Introduction

The integration of 6G Ultra-Reliable Low-Latency Communication (URLLC) technology into
smart manufacturing environments promises to enable real-time control, precise coordination,
and data-driven decision-making. However, the practical impact of 6G network performance
on manufacturing outcomes remains an active area of research.

**Research Question**: Is there a statistically significant and practically meaningful
relationship between 6G network performance metrics (latency, packet loss) and
manufacturing efficiency outcomes (efficiency status, error rates, quality defect rates)?

**Dataset Characteristics**:
- **Rows**: {n_rows:,}
- **Machines**: {n_machines}
- **Time Range**: {date_min} to {date_max} (~69 days)
- **Date Format**: DD-MM-YYYY (explicitly parsed to prevent day/month ambiguity)
- **Columns**: 14 (plus 1 derived column: Network_Quality_Band)

---
"""


def _section_methodology(df: pd.DataFrame) -> str:
    """Generate the methodology section."""
    dist = compute_class_distribution(df, "Efficiency_Status")
    dist_str = ", ".join(f"{k}: {v:.1f}%" for k, v in sorted(dist.items()))
    return f"""## 2. Methodology

### 2.1 Data Preprocessing

The dataset was loaded and validated against the following schema requirements:
- Exactly 100,000 rows and 14 columns
- Required columns present: `Network_Latency_ms`, `Packet_Loss_%`, `Efficiency_Status`,
  `Production_Speed`, `Error_Rate`, `Quality_Control_Defect_Rate`, `Operation_Mode`,
  `Timestamp`, `Machine_ID`
- `Timestamp` parsed using explicit DD-MM-YYYY format: `pd.to_datetime(format='%d-%m-%Y')`
- `Network_Latency_ms` ≥ 0; `Packet_Loss_%` ∈ [0, 100]
- `Efficiency_Status` ∈ {{Low, Medium, High}}; `Operation_Mode` ∈ {{Active, Idle, Maintenance}}

**Efficiency Status Distribution**: {dist_str}

### 2.2 Network Quality Band Classification

Each row is classified into one of three network quality bands based on 6G URLLC requirements:

| Band | Latency Threshold | Packet Loss Threshold |
|------|-------------------|-----------------------|
| High | < 20 ms | < 1% |
| Medium | < 50 ms | < 5% |
| Low | ≥ 50 ms OR ≥ 5% | — |

Both thresholds must be satisfied simultaneously for High and Medium classification.

### 2.3 Key Performance Indicators

#### KPI 1: Network Stability Index (NSI)

Measures the stability of network performance over time.

$$\\text{{NSI}} = 1 - \\frac{{\\text{{CV}}_{{\\text{{latency}}}} + \\text{{CV}}_{{\\text{{packet\\_loss}}}}}}{2}$$

where $\\text{{CV}} = \\min\\left(\\frac{{\\sigma}}{{\\mu}}, 1\\right)$ is the coefficient of variation,
capped at 1.0. NSI ∈ [0, 1], where 1 = perfectly stable.

#### KPI 2: Latency Sensitivity Score (LSS)

Quantifies how manufacturing efficiency changes with network latency using an aggregated
regression approach (row-level correlations ≈ 0).

Method: Encode `Efficiency_Status` as ordinal (Low=1, Medium=2, High=3), bin latency into
10 deciles, compute mean efficiency score per bin, fit linear regression:

$$\\bar{{E}}_i = \\beta_0 + \\beta_1 \\cdot \\bar{{L}}_i + \\epsilon_i, \\quad \\text{{LSS}} = |\\hat{{\\beta}}_1|$$

#### KPI 3: Packet Loss Impact Ratio (PLIR)

Compares mean error rates between high packet-loss (top quartile, Q4) and low packet-loss
(bottom quartile, Q1) groups using Welch's t-test:

$$\\text{{PLIR}} = \\frac{{\\bar{{E}}_{{\\text{{Q4}}}}}}{{\\bar{{E}}_{{\\text{{Q1}}}}}}$$

#### KPI 4: Network-Efficiency Correlation (NEC)

Tests statistical association between network quality band and efficiency class using
a chi-square test of independence and Cramér's V effect size:

$$V = \\sqrt{{\\frac{{\\chi^2}}{{n \\cdot (\\min(r, c) - 1)}}}}$$

where $n$ = sample size, $r$ = number of row categories, $c$ = number of column categories.
V ∈ [0, 1], where 0 = no association, 1 = perfect association.

### 2.4 Statistical Tests

| Method | Use Case |
|--------|----------|
| Chi-square test of independence | Network quality band × Efficiency status |
| Cramér's V | Effect size for chi-square |
| Welch's t-test | Error rate comparison between quartile groups |
| Spearman rank correlation | Non-linear monotonic relationships |
| Aggregated Pearson correlation | Trend analysis on binned data |
| LOWESS smoothing | Visualization of non-linear trends |

---
"""


def _section_results(df: pd.DataFrame, kpis: AllKPIs, fig_dir: Path) -> str:
    """Generate the results section with KPI values and figures."""
    # Generate figures
    eff_fig = _generate_efficiency_distribution_figure(df, fig_dir)
    lat_fig = _generate_latency_distribution_figure(df, fig_dir)
    heatmap_fig = _generate_risk_heatmap_figure(df, fig_dir)

    nsi_val = _fmt(kpis.nsi.nsi, ".3f")
    cv_lat = _fmt(kpis.nsi.cv_latency, ".3f")
    cv_pl = _fmt(kpis.nsi.cv_packet_loss, ".3f")

    lss_val = _fmt(kpis.lss.score, ".4f")
    lss_r2 = _fmt(kpis.lss.r_squared, ".4f")
    lss_p = _fmt(kpis.lss.p_value, ".4f")

    plir_val = _fmt(kpis.plir.ratio, ".3f")
    plir_q1 = _fmt(kpis.plir.q1_error_rate, ".4f")
    plir_q4 = _fmt(kpis.plir.q4_error_rate, ".4f")
    plir_p = _fmt(kpis.plir.p_value, ".4f")

    nec_v = _fmt(kpis.nec.cramers_v, ".3f")
    nec_chi2 = _fmt(kpis.nec.chi2_statistic, ".2f")
    nec_p = _fmt(kpis.nec.p_value, ".4f")

    contingency_md = ""
    if not kpis.nec.contingency_table.empty:
        contingency_md = "\n**Contingency Table (Network_Quality_Band × Efficiency_Status)**:\n\n"
        # Manual markdown table formatting to avoid tabulate dependency
        ct = kpis.nec.contingency_table
        cols = ct.columns.tolist()
        contingency_md += "| Network_Quality_Band | " + " | ".join(str(c) for c in cols) + " |\n"
        contingency_md += "|---|" + "|".join("---" for _ in cols) + "|\n"
        for idx, row in ct.iterrows():
            contingency_md += f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |\n"
        contingency_md += "\n"

    return f"""## 3. Results

### 3.1 Network Stability Index

| Metric | Value |
|--------|-------|
| NSI | **{nsi_val}** |
| CV Latency | {cv_lat} |
| CV Packet Loss | {cv_pl} |

*Interpretation*: {kpis.nsi.interpretation}

### 3.2 Latency Sensitivity Score

| Metric | Value |
|--------|-------|
| LSS (|slope|) | **{lss_val}** |
| R² | {lss_r2} |
| p-value | {lss_p} |

*Interpretation*: {kpis.lss.interpretation}

### 3.3 Packet Loss Impact Ratio

| Metric | Value |
|--------|-------|
| PLIR | **{plir_val}** |
| Q1 Mean Error Rate | {plir_q1} |
| Q4 Mean Error Rate | {plir_q4} |
| Welch's t-test p-value | {plir_p} |

*Interpretation*: {kpis.plir.interpretation}

### 3.4 Network-Efficiency Correlation (Chi-square + Cramér's V)

| Metric | Value |
|--------|-------|
| Cramér's V | **{nec_v}** |
| χ² statistic | {nec_chi2} |
| p-value | {nec_p} |

*Interpretation*: {kpis.nec.interpretation}
{contingency_md}

### 3.5 Visualizations

**Figure 1 — Efficiency Class Distribution by Network Quality Band:**

![Efficiency Distribution]({eff_fig})

**Figure 2 — Network Latency Distribution:**

![Latency Distribution]({lat_fig})

**Figure 3 — Low Efficiency Rate Risk Heatmap:**

![Risk Heatmap]({heatmap_fig})

---
"""


def _section_discussion_limitations_conclusion(kpis: AllKPIs, df: pd.DataFrame) -> str:
    """Generate discussion, recommendations, limitations, and conclusion sections."""
    dist = compute_class_distribution(df, "Efficiency_Status")
    low_pct = dist.get("Low", 0.0)

    nec_significant = (
        not math.isnan(kpis.nec.p_value)
        and kpis.nec.p_value < 0.05
        and not math.isnan(kpis.nec.cramers_v)
        and kpis.nec.cramers_v > 0.1
    )
    plir_meaningful = not math.isnan(kpis.plir.ratio) and kpis.plir.ratio > 1.2
    lss_meaningful = (
        not math.isnan(kpis.lss.score)
        and kpis.lss.score > 0.01
        and not math.isnan(kpis.lss.p_value)
        and kpis.lss.p_value < 0.05
    )

    if nec_significant or plir_meaningful or lss_meaningful:
        finding = (
            "The analysis reveals statistically significant associations between 6G network "
            "performance and manufacturing efficiency outcomes in the Thales dataset."
        )
    else:
        finding = (
            "The analysis does not reveal strong, statistically significant associations between "
            "6G network performance metrics and manufacturing efficiency outcomes at conventional "
            "significance levels. This result is reported transparently and may reflect "
            "characteristics of the dataset rather than the absence of any real-world effect."
        )

    return f"""## 4. Discussion

{finding}

**Key Findings**:
1. **Network Stability (NSI={_fmt(kpis.nsi.nsi, '.3f')})**: {kpis.nsi.interpretation}
2. **Latency Sensitivity (LSS={_fmt(kpis.lss.score, '.4f')})**: {kpis.lss.interpretation}
3. **Packet Loss Impact (PLIR={_fmt(kpis.plir.ratio, '.3f')})**: {kpis.plir.interpretation}
4. **Network-Efficiency Association (V={_fmt(kpis.nec.cramers_v, '.3f')})**: {kpis.nec.interpretation}

**Class Imbalance**: The dataset shows {low_pct:.1f}% of readings classified as Low efficiency.
This heavy imbalance may reduce the statistical power of tests designed to detect differences
across efficiency classes and should be considered when interpreting all KPI values.

---

## 5. Recommendations for 6G Network Configuration

Based on the analysis:

1. **Maintain latency below 20ms** for operations requiring real-time control, consistent
   with High-quality band thresholds derived from 6G URLLC specifications.
2. **Target packet loss below 1%** to minimize data gaps in time-sensitive manufacturing telemetry.
3. **Monitor Network Stability Index** continuously; NSI values below 0.70 warrant investigation
   of network infrastructure for congestion or hardware faults.
4. **Validate with real-world data**: Due to the synthetic characteristics of this dataset,
   findings should be validated against actual manufacturing telemetry before operational deployment.

---

## 6. Limitations

1. **Near-zero row-level correlations**: Row-level Pearson and Spearman correlations between
   network metrics and manufacturing outcomes are approximately zero, suggesting that individual
   telemetry readings do not carry meaningful signal. Aggregated analysis was used to compensate.

2. **Synthetic data characteristics**: The dataset exhibits traits consistent with generated
   or simulated data — clean numeric ranges, no missing values, uniform distributions, and
   round-number values. These characteristics limit the generalizability of findings to
   real-world manufacturing deployments.

3. **Class imbalance**: {low_pct:.1f}% Low efficiency prevalence reduces statistical power
   for detecting efficiency-related effects.

4. **Confounding factors**: Machine type, production batch, operator experience, and other
   unmeasured variables may influence both network conditions and manufacturing efficiency.

5. **Temporal granularity**: Daily-level aggregation used for trend analysis; intra-day
   patterns are not captured.

---

## 7. Conclusion

This study analyzed the statistical relationship between 6G network performance and smart
manufacturing efficiency using {len(df):,} telemetry records from 50 machines.

{finding}

The 6G URLLC threshold-based network quality classification (High/Medium/Low) provides a
useful operational framework for monitoring network health in manufacturing environments.
Future work should validate these findings with real-world manufacturing telemetry data
and incorporate additional contextual variables (machine type, product type, batch size)
to better isolate network-specific effects.

---

## References

1. 3GPP TR 38.824: Study on physical layer enhancements for NR URLLC (Release 16), 2019.
2. ITU-R: Framework and overall objectives of the future development of IMT for 2030 and beyond, 2023.
3. Cramér, H.: Mathematical Methods of Statistics. Princeton University Press, 1946.
4. Cleveland, W.S.: Robust Locally Weighted Regression and Smoothing Scatterplots. JASA, 1979.
5. Thales Group: Manufacturing Telemetry Dataset (Thales_Group_Manufacturing.csv).

---

*Generated by Thales 6G Manufacturing Analytics Platform*  
*Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_research_paper(df: pd.DataFrame, kpis: AllKPIs) -> str:
    """Generate the full technical research paper as a Markdown string.

    This function is the single entry point for report generation. It uses the
    ``kpis`` parameter (produced by ``compute_all_kpis``) as the single source
    of truth for all reported KPI values, ensuring consistency with the dashboard.

    Args:
        df: Validated DataFrame from ``load_and_prepare_dataset()``.
        kpis: AllKPIs result from ``compute_all_kpis(df)``.

    Returns:
        Full research paper as a Markdown-formatted string.

    Side Effects:
        Creates ``reports/figures/`` directory and saves static visualization PNG files.

    Requirements Addressed:
        - 15.1–15.9: Research paper structure and content
        - 22.1–22.5: Synthetic data transparency
        - 23.5: KPI values from single source of truth
    """
    fig_dir = _ensure_figures_dir()

    sections = [
        _section_title_abstract(kpis),
        _section_introduction(df),
        _section_methodology(df),
        _section_results(df, kpis, fig_dir),
        _section_discussion_limitations_conclusion(kpis, df),
    ]

    return "\n".join(sections)
