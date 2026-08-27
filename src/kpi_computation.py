"""
Shared KPI Computation Module for Thales 6G Manufacturing Analytics Platform.

This module is the **single source of truth** for all KPI computations. Both the
Streamlit dashboard and the report generation pipeline call these functions directly,
ensuring that KPI values are always consistent across all outputs.

KPIs Implemented:
    1. Network Stability Index (NSI)  — measures temporal stability of network performance
    2. Latency Sensitivity Score (LSS) — quantifies manufacturing efficiency change with latency
    3. Packet Loss Impact Ratio (PLIR) — compares error rates between high/low packet loss groups
    4. Network-Efficiency Correlation (NEC) — chi-square association between network quality and efficiency

Design Principles:
    - All KPIs accept a filtered DataFrame so they recompute correctly on any subset
    - Aggregated (binned) analysis is used throughout, because row-level correlations are ~0
    - Edge cases (zero denominators, insufficient data) return np.nan with a descriptive note
    - Streamlit caching is applied to the unified compute_all_kpis() function

Requirements Addressed:
    - 3.1–3.5: Network Stability Index
    - 4.1–4.5: Latency Sensitivity Score
    - 5.1–5.5: Packet Loss Impact Ratio
    - 6.1–6.6: Network-Efficiency Correlation
    - 20.2, 20.3, 20.6: Docstrings and type hints
    - 23.5: Single source of truth for KPI values
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import os as _os

# Only import Streamlit (and its slow runtime initialization) when actually running
# inside a Streamlit server process. In pytest / CLI contexts, use a no-op decorator.
if _os.environ.get("STREAMLIT_SERVER_PORT") or _os.environ.get("STREAMLIT_RUN_TARGET"):
    import streamlit as st
    _cache_data = st.cache_data
else:
    import functools as _functools
    def _cache_data(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            @_functools.wraps(fn)
            def wrapper(*a, **kw):
                return fn(*a, **kw)
            return wrapper
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NetworkStabilityResult:
    """Result container for the Network Stability Index (NSI).

    Attributes:
        nsi: Stability index in [0, 1].  1 = perfectly stable, 0 = extremely variable.
        cv_latency: Coefficient of variation for Network_Latency_ms (capped at 1.0).
        cv_packet_loss: Coefficient of variation for Packet_Loss_% (capped at 1.0).
        interpretation: Plain-language description of the NSI value.
    """

    nsi: float
    cv_latency: float
    cv_packet_loss: float
    interpretation: str


@dataclass
class LatencySensitivityResult:
    """Result container for the Latency Sensitivity Score (LSS).

    Attributes:
        score: Absolute value of the linear regression slope (efficiency score / ms).
        r_squared: Coefficient of determination for the aggregated regression.
        p_value: Two-tailed p-value for the slope coefficient.
        interpretation: Plain-language description of the LSS value.
        bin_data: DataFrame with decile midpoints and mean efficiency scores.
    """

    score: float
    r_squared: float
    p_value: float
    interpretation: str
    bin_data: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PacketLossImpactResult:
    """Result container for the Packet Loss Impact Ratio (PLIR).

    Attributes:
        ratio: Q4_error_rate / Q1_error_rate.  np.nan if Q1 mean is zero.
        q1_error_rate: Mean Error_Rate for bottom-quartile (low) packet loss group.
        q4_error_rate: Mean Error_Rate for top-quartile (high) packet loss group.
        p_value: Welch's t-test p-value.  np.nan if test is not computable.
        interpretation: Plain-language description of the PLIR value.
    """

    ratio: float
    q1_error_rate: float
    q4_error_rate: float
    p_value: float
    interpretation: str


@dataclass
class NetworkEfficiencyCorrelationResult:
    """Result container for the Network-Efficiency Correlation (NEC) via chi-square.

    Attributes:
        cramers_v: Effect size in [0, 1].  0 = no association, 1 = perfect association.
        chi2_statistic: Chi-square test statistic.
        p_value: Chi-square test p-value.
        contingency_table: Cross-tabulation of Network_Quality_Band × Efficiency_Status.
        interpretation: Plain-language description of the NEC value.
    """

    cramers_v: float
    chi2_statistic: float
    p_value: float
    contingency_table: pd.DataFrame
    interpretation: str


@dataclass
class AllKPIs:
    """Container holding all four computed KPI results.

    This is the canonical output of compute_all_kpis() and is consumed by both
    the Streamlit dashboard and the report generation pipeline.

    Attributes:
        nsi: Network Stability Index result.
        lss: Latency Sensitivity Score result.
        plir: Packet Loss Impact Ratio result.
        nec: Network-Efficiency Correlation result.
    """

    nsi: NetworkStabilityResult
    lss: LatencySensitivityResult
    plir: PacketLossImpactResult
    nec: NetworkEfficiencyCorrelationResult


# ---------------------------------------------------------------------------
# KPI 1 — Network Stability Index
# ---------------------------------------------------------------------------


def compute_network_stability_index(df: pd.DataFrame) -> NetworkStabilityResult:
    """Compute the Network Stability Index (NSI) for a given dataset.

    The NSI measures how stable network performance is over time.  It is derived
    from the coefficient of variation (CV = std / mean) for both latency and
    packet loss, then transformed so that higher values represent better stability.

    Mathematical Formula (LaTeX):
        $$\\text{NSI} = 1 - \\frac{\\text{CV}_{\\text{latency}} + \\text{CV}_{\\text{packet\\_loss}}}{2}$$

    where:
        $$\\text{CV} = \\min\\left(\\frac{\\sigma}{\\mu}, 1\\right)$$

    Both CVs are capped at 1.0 to ensure the NSI stays in [0, 1] even for
    extreme variance.  If the mean of a metric is zero (perfectly absent), the
    CV is set to 0 (treating absence as perfectly stable).

    Args:
        df: DataFrame containing at minimum:
            - ``Network_Latency_ms`` (numeric, non-negative)
            - ``Packet_Loss_%`` (numeric, in [0, 100])

    Returns:
        NetworkStabilityResult with NSI in [0, 1] and supporting diagnostics.

    Raises:
        ValueError: If required columns are missing from ``df``.

    Requirements Addressed:
        - 3.1: Compute NSI from latency and packet loss variability
        - 3.2: NSI formula as documented
        - 3.3: NSI is bounded [0, 1]
        - 3.4: NSI exposed as reusable function
        - 3.5: Edge case handled (zero mean → CV = 0)
    """
    required = {"Network_Latency_ms", "Packet_Loss_%"}
    _check_required_columns(df, required, "compute_network_stability_index")

    lat = df["Network_Latency_ms"]
    pl = df["Packet_Loss_%"]

    cv_lat = _safe_cv(lat)
    cv_pl = _safe_cv(pl)

    nsi = float(1.0 - (cv_lat + cv_pl) / 2.0)
    # Clamp to [0, 1] for floating-point safety
    nsi = max(0.0, min(1.0, nsi))

    interpretation = _interpret_nsi(nsi)
    return NetworkStabilityResult(
        nsi=nsi,
        cv_latency=cv_lat,
        cv_packet_loss=cv_pl,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# KPI 2 — Latency Sensitivity Score
# ---------------------------------------------------------------------------


def compute_latency_sensitivity_score(df: pd.DataFrame) -> LatencySensitivityResult:
    """Compute the Latency Sensitivity Score (LSS).

    LSS quantifies how much manufacturing efficiency degrades as network latency
    increases.  Because row-level correlations are approximately zero for this
    dataset, an *aggregated* approach is used:

    Method:
        1. Encode ``Efficiency_Status`` as an ordinal score (Low=1, Medium=2, High=3).
        2. Bin ``Network_Latency_ms`` into 10 equal-frequency deciles (pd.qcut).
        3. Compute the mean efficiency score and mean latency midpoint per bin.
        4. Fit a linear regression: efficiency_score ~ latency_midpoint using
           ``scipy.stats.linregress``.
        5. The LSS is the absolute value of the slope (sensitivity per ms).

    Mathematical Notation (LaTeX):
        $$\\text{LSS} = |\\hat{\\beta}_1|$$

    where :math:`\\hat{\\beta}_1` is the slope of:
        $$\\bar{E}_i = \\beta_0 + \\beta_1 \\cdot \\bar{L}_i + \\epsilon_i$$

    and :math:`\\bar{E}_i`, :math:`\\bar{L}_i` are mean efficiency score and
    mean latency for decile bin :math:`i`.

    Args:
        df: DataFrame containing at minimum:
            - ``Network_Latency_ms`` (numeric)
            - ``Efficiency_Status`` (categorical: Low/Medium/High)

    Returns:
        LatencySensitivityResult with score, r_squared, p_value, interpretation,
        and the aggregated bin_data DataFrame.

    Raises:
        ValueError: If required columns are missing or fewer than 10 rows exist.

    Requirements Addressed:
        - 4.1: LSS computed via decile binning and aggregated regression
        - 4.2: Returns dict-compatible result with score, r_squared, p_value
        - 4.3: Aggregated regression approach documented
        - 4.4: LSS exposed as reusable function
        - 4.5: Edge case handled (<10 rows)
    """
    required = {"Network_Latency_ms", "Efficiency_Status"}
    _check_required_columns(df, required, "compute_latency_sensitivity_score")

    if len(df) < 10:
        return LatencySensitivityResult(
            score=float("nan"),
            r_squared=float("nan"),
            p_value=float("nan"),
            interpretation="Insufficient data for LSS computation (< 10 rows).",
        )

    work = df[["Network_Latency_ms", "Efficiency_Status"]].copy()
    work["efficiency_score"] = work["Efficiency_Status"].map(
        {"Low": 1, "Medium": 2, "High": 3}
    )
    work = work.dropna(subset=["efficiency_score"])

    try:
        work["latency_bin"] = pd.qcut(
            work["Network_Latency_ms"], q=10, duplicates="drop"
        )
    except ValueError:
        # Fallback: fewer unique values than bins
        return LatencySensitivityResult(
            score=float("nan"),
            r_squared=float("nan"),
            p_value=float("nan"),
            interpretation="Insufficient latency variance for LSS computation.",
        )

    agg = (
        work.groupby("latency_bin", observed=True)
        .agg(
            mean_latency=("Network_Latency_ms", "mean"),
            mean_efficiency=("efficiency_score", "mean"),
        )
        .dropna()
        .reset_index()
    )

    if len(agg) < 2:
        return LatencySensitivityResult(
            score=float("nan"),
            r_squared=float("nan"),
            p_value=float("nan"),
            interpretation="Insufficient distinct latency bins for LSS computation.",
        )

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        agg["mean_latency"], agg["mean_efficiency"]
    )
    r_squared = float(r_value**2)
    score = float(abs(slope))

    interpretation = _interpret_lss(score, p_value)
    return LatencySensitivityResult(
        score=score,
        r_squared=r_squared,
        p_value=float(p_value),
        interpretation=interpretation,
        bin_data=agg,
    )


# ---------------------------------------------------------------------------
# KPI 3 — Packet Loss Impact Ratio
# ---------------------------------------------------------------------------


def compute_packet_loss_impact_ratio(df: pd.DataFrame) -> PacketLossImpactResult:
    """Compute the Packet Loss Impact Ratio (PLIR).

    PLIR compares the manufacturing error rate between machines experiencing
    high packet loss versus those experiencing low packet loss, using quartile
    grouping and Welch's independent t-test.

    Method:
        1. Compute the 25th percentile (Q1) and 75th percentile (Q3) of
           ``Packet_Loss_%`` across the filtered dataset.
        2. Define two groups:
           - **Low packet loss group**: rows where ``Packet_Loss_%`` <= Q1
           - **High packet loss group**: rows where ``Packet_Loss_%`` >= Q3
        3. Compute mean ``Error_Rate`` for each group.
        4. Compute the ratio: PLIR = high_group_mean / low_group_mean.
        5. Perform Welch's t-test (``scipy.stats.ttest_ind`` with ``equal_var=False``).

    Mathematical Notation (LaTeX):
        $$\\text{PLIR} = \\frac{\\bar{E}_{\\text{Q4}}}{\\bar{E}_{\\text{Q1}}}$$

    Args:
        df: DataFrame containing at minimum:
            - ``Packet_Loss_%`` (numeric, in [0, 100])
            - ``Error_Rate`` (numeric)

    Returns:
        PacketLossImpactResult with ratio, group means, p_value, and interpretation.
        ``ratio`` and ``p_value`` are ``np.nan`` if computation is not possible.

    Raises:
        ValueError: If required columns are missing.

    Requirements Addressed:
        - 5.1: PLIR computed via quartile grouping
        - 5.2: PLIR is non-negative or NaN
        - 5.3: Welch's t-test applied
        - 5.4: PLIR exposed as reusable function
        - 5.5: Zero denominator handled with np.nan
    """
    required = {"Packet_Loss_%", "Error_Rate"}
    _check_required_columns(df, required, "compute_packet_loss_impact_ratio")

    q1_threshold = df["Packet_Loss_%"].quantile(0.25)
    q3_threshold = df["Packet_Loss_%"].quantile(0.75)

    low_group = df.loc[df["Packet_Loss_%"] <= q1_threshold, "Error_Rate"].dropna()
    high_group = df.loc[df["Packet_Loss_%"] >= q3_threshold, "Error_Rate"].dropna()

    q1_mean = float(low_group.mean()) if len(low_group) > 0 else float("nan")
    q4_mean = float(high_group.mean()) if len(high_group) > 0 else float("nan")

    # Handle zero denominator
    if math.isnan(q1_mean) or q1_mean == 0.0:
        ratio = float("nan")
        p_value = float("nan")
        interpretation = (
            "PLIR could not be computed: mean error rate in low packet-loss group is zero."
        )
    elif math.isnan(q4_mean):
        ratio = float("nan")
        p_value = float("nan")
        interpretation = "PLIR could not be computed: insufficient data in high packet-loss group."
    else:
        ratio = float(q4_mean / q1_mean)

        if len(low_group) >= 2 and len(high_group) >= 2:
            t_stat, p_value = stats.ttest_ind(high_group, low_group, equal_var=False)
            p_value = float(p_value)
        else:
            p_value = float("nan")

        interpretation = _interpret_plir(ratio, p_value)

    return PacketLossImpactResult(
        ratio=ratio,
        q1_error_rate=q1_mean,
        q4_error_rate=q4_mean,
        p_value=p_value,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# KPI 4 — Network-Efficiency Correlation (Chi-square + Cramér's V)
# ---------------------------------------------------------------------------


def compute_network_efficiency_correlation(
    df: pd.DataFrame,
) -> NetworkEfficiencyCorrelationResult:
    """Compute Network-Efficiency Correlation (NEC) via chi-square test.

    This KPI tests whether there is a statistically significant association
    between network quality band and manufacturing efficiency class using a
    chi-square test of independence.  The effect size is measured by Cramér's V.

    Method:
        1. Build a contingency table: rows = Network_Quality_Band, cols = Efficiency_Status.
        2. Apply ``scipy.stats.chi2_contingency`` with ``correction=False``.
        3. Compute Cramér's V effect size:
           $$V = \\sqrt{\\frac{\\chi^2}{n \\cdot (\\min(r, c) - 1)}}$$
           where :math:`n` = sample size, :math:`r` = number of rows, :math:`c` = number of columns.

    Args:
        df: DataFrame containing at minimum:
            - ``Network_Quality_Band`` (categorical: Low/Medium/High)
            - ``Efficiency_Status`` (categorical: Low/Medium/High)

    Returns:
        NetworkEfficiencyCorrelationResult with cramers_v, chi2_statistic, p_value,
        contingency_table, and interpretation.

    Raises:
        ValueError: If required columns are missing or contingency table has fewer
                    than 2 rows or 2 columns.

    Requirements Addressed:
        - 6.1: Contingency table (Network_Quality_Band × Efficiency_Status)
        - 6.2: Chi-square test applied
        - 6.3: Cramér's V in [0, 1]
        - 6.4: NEC exposed as reusable function
        - 6.5: Edge case handled (degenerate table)
        - 6.6: Interpretation generated
    """
    required = {"Network_Quality_Band", "Efficiency_Status"}
    _check_required_columns(df, required, "compute_network_efficiency_correlation")

    contingency = pd.crosstab(df["Network_Quality_Band"], df["Efficiency_Status"])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        empty_ct = pd.DataFrame()
        return NetworkEfficiencyCorrelationResult(
            cramers_v=float("nan"),
            chi2_statistic=float("nan"),
            p_value=float("nan"),
            contingency_table=empty_ct,
            interpretation="NEC could not be computed: contingency table has fewer than 2 rows or columns.",
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(
        contingency, correction=False
    )
    n = int(contingency.values.sum())
    min_dim = min(contingency.shape) - 1

    cramers_v = float(math.sqrt(chi2 / (n * min_dim))) if (n > 0 and min_dim > 0) else float("nan")
    # Clamp to [0, 1] for numerical safety
    cramers_v = max(0.0, min(1.0, cramers_v))

    interpretation = _interpret_nec(cramers_v, float(p_value))
    return NetworkEfficiencyCorrelationResult(
        cramers_v=cramers_v,
        chi2_statistic=float(chi2),
        p_value=float(p_value),
        contingency_table=contingency,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Unified KPI function — single source of truth
# ---------------------------------------------------------------------------


@_cache_data(ttl=3600, max_entries=50)
def compute_all_kpis(df: pd.DataFrame) -> AllKPIs:
    """Compute all four KPIs from a (filtered) DataFrame.

    This is the **single source of truth** for KPI computation, called by both
    the Streamlit dashboard and the report generation pipeline.  Using this
    function ensures that KPI values are identical across all outputs.

    Caching:
        Decorated with ``@st.cache_data(ttl=3600, max_entries=50)`` so that
        repeated calls with the same DataFrame (e.g., no filter change) return
        the cached result without recomputation.

    Args:
        df: Filtered or full DataFrame from ``load_and_prepare_dataset()``.
            Must contain all columns required by each KPI function.

    Returns:
        AllKPIs dataclass instance containing NSI, LSS, PLIR, and NEC results.

    Requirements Addressed:
        - 3.4, 4.4, 5.4, 6.4: Each KPI exposed via unified function
        - 23.5: Single source of truth pattern
    """
    nsi = compute_network_stability_index(df)
    lss = compute_latency_sensitivity_score(df)
    plir = compute_packet_loss_impact_ratio(df)
    nec = compute_network_efficiency_correlation(df)
    return AllKPIs(nsi=nsi, lss=lss, plir=plir, nec=nec)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_required_columns(
    df: pd.DataFrame, required: set, fn_name: str
) -> None:
    """Raise ValueError if any required columns are missing from df."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{fn_name}: missing required column(s): {', '.join(sorted(missing))}"
        )


def _safe_cv(series: pd.Series) -> float:
    """Return coefficient of variation capped at 1.0; returns 0.0 if mean is zero."""
    mean = series.mean()
    std = series.std()
    if mean == 0 or math.isnan(mean):
        return 0.0
    cv = std / mean
    return float(min(cv, 1.0))


def _interpret_nsi(nsi: float) -> str:
    """Return a plain-language interpretation of the NSI value."""
    if nsi >= 0.85:
        return f"Network performance is highly stable (NSI={nsi:.3f}). Low variability in latency and packet loss."
    elif nsi >= 0.70:
        return f"Network performance is moderately stable (NSI={nsi:.3f}). Some variability present."
    else:
        return (
            f"Network performance shows notable instability (NSI={nsi:.3f}). "
            "High variability in latency or packet loss may impact manufacturing reliability."
        )


def _interpret_lss(score: float, p_value: float) -> str:
    """Return a plain-language interpretation of the LSS value."""
    if math.isnan(score):
        return "LSS could not be computed."
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    if score > 0.01:
        return (
            f"Measurable latency sensitivity detected (LSS={score:.4f}, p={p_value:.4f}). "
            f"Efficiency changes {significance}ly with latency."
        )
    else:
        return (
            f"Minimal latency sensitivity observed (LSS={score:.4f}, p={p_value:.4f}). "
            f"Efficiency appears {significance}ly related to latency at the aggregated level."
        )


def _interpret_plir(ratio: float, p_value: float) -> str:
    """Return a plain-language interpretation of the PLIR value."""
    if math.isnan(ratio):
        return "PLIR could not be computed."
    significance = "statistically significant" if (not math.isnan(p_value) and p_value < 0.05) else "not statistically significant"
    if ratio > 1.2:
        return (
            f"High packet loss is associated with {ratio:.2f}× higher error rates (PLIR={ratio:.2f}, p={p_value:.4f}). "
            f"The difference is {significance}."
        )
    elif ratio > 1.0:
        return (
            f"Slight increase in error rates with high packet loss (PLIR={ratio:.2f}, p={p_value:.4f}). "
            f"The difference is {significance}."
        )
    else:
        return (
            f"No meaningful increase in error rates with high packet loss (PLIR={ratio:.2f}, p={p_value:.4f}). "
            f"The difference is {significance}."
        )


def _interpret_nec(cramers_v: float, p_value: float) -> str:
    """Return a plain-language interpretation of the NEC (Cramér's V) value."""
    if math.isnan(cramers_v):
        return "NEC could not be computed."
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    if cramers_v > 0.3:
        strength = "strong"
    elif cramers_v > 0.1:
        strength = "moderate"
    else:
        strength = "weak"
    return (
        f"A {strength} {significance} association between network quality and efficiency "
        f"was found (Cramér's V={cramers_v:.3f}, p={p_value:.4f})."
    )
