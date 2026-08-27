"""
Executive Summary Generator for Thales 6G Manufacturing Analytics Platform.

This module generates a concise, non-technical executive summary (≤800 words / ~2 pages)
from the full research paper and KPI results. It is designed for government and business
stakeholders who need actionable insights without statistical jargon.

Usage:
    from executive_summary_generator import generate_executive_summary
    summary_md = generate_executive_summary(paper, kpis, df)

Requirements Addressed:
    - 16.1–16.8: Executive summary generation
    - 22.1–22.5: Synthetic data transparency
    - 20.3, 20.6: Docstrings and type hints
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kpi_computation import AllKPIs
from statistical_analysis import compute_class_distribution

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def _fmt(value: float, fmt: str = ".2f") -> str:
    if math.isnan(value):
        return "N/A"
    return format(value, fmt)


def _nsi_to_business(nsi: float) -> str:
    """Translate NSI to plain business language."""
    if math.isnan(nsi):
        return "Network stability could not be measured."
    pct = int(min(100, max(0, nsi * 100)))
    if nsi >= 0.85:
        return f"Network performance is highly consistent ({pct}% stable). Variability is low."
    elif nsi >= 0.70:
        return f"Network performance is moderately consistent ({pct}% stable). Some variability is present."
    else:
        return (
            f"Network performance shows notable inconsistency ({pct}% stable). "
            "High variability may affect manufacturing reliability."
        )


def _plir_to_business(plir: float) -> str:
    """Translate PLIR to plain business language."""
    if math.isnan(plir):
        return "The impact of packet loss on manufacturing errors could not be measured."
    if plir > 1.2:
        return (
            f"Factories with poor network packet delivery experienced {plir:.1f}× higher "
            "manufacturing error rates compared to factories with reliable delivery."
        )
    elif plir > 1.0:
        return (
            "Factories with poor packet delivery showed slightly higher error rates, "
            "but the difference was small."
        )
    else:
        return "No meaningful difference in error rates was found between network quality groups."


def _nec_to_business(cramers_v: float, p_value: float) -> str:
    """Translate NEC to plain business language."""
    if math.isnan(cramers_v) or math.isnan(p_value):
        return "The relationship between network quality and efficiency could not be measured."
    if p_value < 0.05 and cramers_v > 0.1:
        return (
            "A meaningful link between network quality and manufacturing efficiency was identified. "
            "Factories with better network conditions tended to show higher efficiency."
        )
    elif p_value < 0.05:
        return (
            "A statistically detectable but practically small association between network quality "
            "and efficiency was found."
        )
    else:
        return (
            "No statistically meaningful link between network quality and manufacturing efficiency "
            "was found in this dataset."
        )


def _generate_exec_bar_chart(df: pd.DataFrame, fig_dir: Path) -> str:
    """Generate a simple efficiency distribution bar chart for non-technical audience."""
    fig_path = fig_dir / "exec_efficiency_bar.png"
    ct = pd.crosstab(df["Network_Quality_Band"], df["Efficiency_Status"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"}
    bottom = np.zeros(len(ct))
    bands = ct.index.tolist()

    for col in ["Low", "Medium", "High"]:
        if col in ct.columns:
            vals = ct[col].values
            ax.bar(bands, vals, bottom=bottom, label=col, color=colors.get(col, "#aaa"), width=0.5)
            bottom += vals

    ax.set_xlabel("Network Quality", fontsize=11)
    ax.set_ylabel("Proportion (%)", fontsize=11)
    ax.set_title("Manufacturing Efficiency by Network Quality", fontsize=12)
    ax.legend(title="Efficiency", fontsize=9)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path.relative_to(fig_path.parent.parent))


def generate_executive_summary(paper: str, kpis: AllKPIs, df: pd.DataFrame) -> str:
    """Generate a concise executive summary (≤800 words) for non-technical stakeholders.

    The summary translates technical KPI values into plain business language and
    provides up to 5 actionable recommendations. It does not use statistical jargon
    (no "Cramér's V", "p-value", "chi-square", etc.).

    Args:
        paper: Full research paper Markdown string (from ``generate_research_paper``).
               Used only for context; KPI values come from the ``kpis`` parameter.
        kpis: AllKPIs result from ``compute_all_kpis(df)`` — single source of truth.
        df: Validated full DataFrame from ``load_and_prepare_dataset()``.

    Returns:
        Executive summary as a Markdown-formatted string (≤800 words, ~2 pages).

    Side Effects:
        Creates ``reports/figures/`` directory and saves summary visualization PNG files.

    Requirements Addressed:
        - 16.1: Executive summary generated
        - 16.2: Plain language (no jargon)
        - 16.3: ≤2 pages / ~800 words
        - 16.4: Business-friendly KPI translations
        - 16.5: ≤5 actionable recommendations
        - 16.6: Caveats about data characteristics
        - 16.7: No statistical jargon
        - 16.8: ≤2 simple visualizations
        - 22.5: Caveat about synthetic data
    """
    fig_dir = _ensure_figures_dir()

    # Generate simplified figure
    bar_fig = _generate_exec_bar_chart(df, fig_dir)

    # Dataset summary
    n_rows = len(df)
    n_machines = df["Machine_ID"].nunique() if "Machine_ID" in df.columns else "N/A"
    dist = compute_class_distribution(df, "Efficiency_Status")
    low_pct = dist.get("Low", 0.0)

    # Business-language translations
    nsi_biz = _nsi_to_business(kpis.nsi.nsi)
    plir_biz = _plir_to_business(kpis.plir.ratio)
    nec_biz = _nec_to_business(kpis.nec.cramers_v, kpis.nec.p_value)

    # Recommendations (≤5)
    recs: List[str] = []

    if not math.isnan(kpis.nsi.nsi) and kpis.nsi.nsi < 0.70:
        recs.append(
            "**Improve network reliability**: Network performance variability is high. "
            "Conduct an audit of network hardware and connectivity to reduce disruptions."
        )

    if not math.isnan(kpis.plir.ratio) and kpis.plir.ratio > 1.2:
        recs.append(
            f"**Reduce packet loss**: Factories with poor packet delivery experienced "
            f"{kpis.plir.ratio:.1f}× higher error rates. Target ≥99% packet delivery reliability."
        )

    if (
        not math.isnan(kpis.nec.cramers_v)
        and not math.isnan(kpis.nec.p_value)
        and kpis.nec.p_value < 0.05
        and kpis.nec.cramers_v > 0.1
    ):
        recs.append(
            "**Prioritize High-quality network conditions**: Data suggests factories operating "
            "on high-quality networks tend to achieve better efficiency outcomes."
        )

    recs.append(
        "**Establish continuous monitoring**: Deploy real-time network quality dashboards "
        "to identify degradation events before they impact production."
    )

    recs.append(
        "**Validate with production data**: These findings are based on historical simulation "
        "data. Validate conclusions with live manufacturing telemetry before implementing "
        "infrastructure changes."
    )

    recs = recs[:5]  # Cap at 5
    recs_md = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs))

    return f"""# Executive Summary: 6G Network Performance & Manufacturing Efficiency

**Prepared for**: Government and Industry Stakeholders  
**Date**: {datetime.now().strftime("%B %Y")}  
**Dataset**: {n_rows:,} telemetry records from {n_machines} industrial machines (~69 days)

---

## Overview

This report summarizes findings from an analysis of 6G wireless network performance and
its relationship to smart manufacturing efficiency. The goal was to determine whether
improvements to network infrastructure — specifically reducing delays and data losses —
translate into better manufacturing outcomes.

---

## What We Measured

We analyzed four aspects of network and manufacturing performance:

1. **Network Consistency**: How stable and predictable is the network day-to-day?  
   → {nsi_biz}

2. **Delay Impact on Efficiency**: Does higher network delay reduce manufacturing efficiency?  
   → {kpis.lss.interpretation}

3. **Data Loss Impact on Errors**: Do factories with more lost data packets make more errors?  
   → {plir_biz}

4. **Overall Network-Efficiency Link**: Is there a clear relationship between network quality and efficiency?  
   → {nec_biz}

---

## Key Finding

> **{low_pct:.0f}%** of all manufacturing readings in the dataset recorded **Low efficiency**.

This high proportion of low-efficiency readings is a notable pattern. Whether this reflects
real operating conditions or characteristics of the data source is an important open question
(see Caveats section below).

---

## Visualisation

**Manufacturing Efficiency by Network Quality Band:**

![Efficiency by Network Quality]({bar_fig})

---

## Recommendations

{recs_md}

---

## Caveats

> [!IMPORTANT]
> The dataset used for this analysis exhibits characteristics consistent with simulated or
> generated data: clean numeric ranges, no missing values, and uniform statistical distributions.
> Findings should be treated as **indicative**, not definitive. We strongly recommend validating
> all conclusions against live manufacturing telemetry before making infrastructure investment decisions.

Additional limitations:
- The data does not capture machine type, product batch, or operator-level factors that may
  also influence efficiency outcomes.
- Analysis covers approximately 69 days; seasonal or long-term trends are not captured.

---

*Generated by Thales 6G Manufacturing Analytics Platform · {datetime.now().strftime("%Y-%m-%d")}*  
*Full technical details available in the Research Paper: `reports/research_paper.md`*
"""
