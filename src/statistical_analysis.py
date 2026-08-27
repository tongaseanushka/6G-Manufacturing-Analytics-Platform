"""
Statistical Analysis Module for Thales 6G Manufacturing Analytics Platform.

This module provides reusable statistical analysis functions used by both the
Streamlit dashboard and the report generation pipeline.  All functions are
designed to handle the specific characteristics of the Thales dataset: large
sample size (100k rows), near-zero row-level correlations, and heavy class
imbalance (≈77.8% Low efficiency).

Functions:
    compute_spearman_correlation  — non-parametric rank correlation
    apply_lowess_smoothing        — LOWESS trend smoothing for scatter plots
    compute_aggregated_pearson    — Pearson correlation on binned/aggregated data
    compute_class_distribution    — percentage distribution of categorical column

Design Notes:
    - Row-level correlations are approximately zero in this dataset, so all
      regression-based analysis is performed on binned/aggregated data.
    - LOWESS smoothing is used exclusively for visualization; it is not used as
      a predictive model.
    - Spearman correlation is used for non-linear monotonic relationships.

Requirements Addressed:
    - 17.1: Statistical analysis module
    - 17.3: Spearman correlation
    - 17.4: LOWESS smoothing
    - 17.5: Edge cases handled (constant columns)
    - 17.6: Aggregated Pearson correlation
    - 21.1: Class distribution percentages
    - 20.2, 20.3, 20.6: Docstrings and type hints
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.nonparametric.smoothers_lowess as sm_lowess


def compute_spearman_correlation(
    df: pd.DataFrame, var1: str, var2: str
) -> Dict[str, float]:
    """Compute Spearman rank correlation between two columns.

    Spearman correlation is appropriate for detecting monotonic (but not
    necessarily linear) relationships between variables.  It is more robust than
    Pearson when distributions are skewed or contain outliers — both common in
    manufacturing telemetry data.

    NaN values in either column are excluded from the computation
    (``nan_policy='omit'``).

    Args:
        df: DataFrame containing both variables.
        var1: Name of the first column (independent variable).
        var2: Name of the second column (dependent variable).

    Returns:
        Dict with keys:
            - ``'rho'``: Spearman correlation coefficient in [-1, 1].
              Returns ``np.nan`` if the column is constant.
            - ``'p_value'``: Two-tailed p-value in [0, 1].
              Returns ``np.nan`` if the column is constant.

    Raises:
        ValueError: If ``var1`` or ``var2`` are not columns in ``df``.

    Examples:
        >>> result = compute_spearman_correlation(df, 'Network_Latency_ms', 'Error_Rate')
        >>> print(result['rho'], result['p_value'])

    Requirements Addressed:
        - 17.1: Statistical analysis function
        - 17.3: Spearman correlation
        - 17.5: Handles constant columns (returns np.nan)
    """
    for col in (var1, var2):
        if col not in df.columns:
            raise ValueError(f"compute_spearman_correlation: column '{col}' not found in DataFrame.")

    series1 = df[var1].dropna()
    series2 = df[var2].dropna()

    # Align on common index
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx]
    s2 = series2.loc[common_idx]

    if len(s1) < 3:
        return {"rho": float("nan"), "p_value": float("nan")}

    # Constant column check — spearmanr returns nan for constant inputs
    if s1.std() == 0 or s2.std() == 0:
        return {"rho": float("nan"), "p_value": float("nan")}

    result = stats.spearmanr(s1, s2, nan_policy="omit")
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def apply_lowess_smoothing(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    frac: float = 0.1,
) -> pd.DataFrame:
    """Apply LOWESS (Locally Weighted Scatterplot Smoothing) to a pair of columns.

    LOWESS produces a smooth non-parametric trend curve, useful for visualizing
    the underlying relationship between two variables in noisy scatter plots.
    It does not assume a parametric functional form, making it suitable for
    exploratory analysis of manufacturing telemetry data.

    This function is intended for **visualization only** and should not be used
    as a predictive model.

    Args:
        df: DataFrame containing the x and y columns.
        x_col: Name of the independent (x-axis) column.
        y_col: Name of the dependent (y-axis) column.
        frac: Fraction of data used for each local regression window.
              Smaller values → less smoothing; larger values → more smoothing.
              Default 0.1 (10% of data per window) works well for 100k rows.

    Returns:
        DataFrame with two columns:
            - ``'x_smoothed'``: Sorted x values used for smoothing.
            - ``'y_smoothed'``: Corresponding smoothed y values.

    Raises:
        ValueError: If ``x_col`` or ``y_col`` are not columns in ``df``,
                    or if fewer than 3 non-null paired rows exist.

    Examples:
        >>> smooth = apply_lowess_smoothing(df, 'Network_Latency_ms', 'Error_Rate')
        >>> plt.plot(smooth['x_smoothed'], smooth['y_smoothed'])

    Requirements Addressed:
        - 17.1: Statistical analysis function
        - 17.4: LOWESS smoothing
        - 17.5: Handles insufficient data (<3 points)
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"apply_lowess_smoothing: column '{col}' not found in DataFrame.")

    work = df[[x_col, y_col]].dropna()

    if len(work) < 3:
        raise ValueError(
            f"apply_lowess_smoothing: insufficient data — need at least 3 non-null rows, "
            f"got {len(work)}."
        )

    smoothed = sm_lowess.lowess(
        work[y_col].values,
        work[x_col].values,
        frac=frac,
        it=3,
        return_sorted=True,
    )

    return pd.DataFrame(
        {"x_smoothed": smoothed[:, 0], "y_smoothed": smoothed[:, 1]}
    )


def compute_aggregated_pearson(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = 10,
) -> Dict:
    """Compute Pearson correlation on binned (aggregated) data.

    Because row-level Pearson correlations are approximately zero in the Thales
    dataset (consistent with a synthetic or heavily averaged dataset), this
    function bins the data and computes the correlation on the bin means.
    This approach reveals trends that are obscured by row-level noise.

    Method:
        1. Bin ``x_col`` into ``n_bins`` equal-frequency bins using ``pd.qcut``.
        2. Compute the mean of ``x_col`` and ``y_col`` for each bin.
        3. Compute Pearson correlation on the resulting aggregated data using
           ``scipy.stats.pearsonr``.

    Mathematical Notation (LaTeX):
        $$r = \\text{Pearson}(\\bar{x}_1, \\ldots, \\bar{x}_k; \\bar{y}_1, \\ldots, \\bar{y}_k)$$

    Args:
        df: DataFrame containing the x and y columns.
        x_col: Name of the independent (x-axis) column to bin.
        y_col: Name of the dependent (y-axis) column.
        n_bins: Number of equal-frequency bins (default 10).

    Returns:
        Dict with keys:
            - ``'r'``: Pearson correlation coefficient in [-1, 1].
            - ``'p_value'``: Two-tailed p-value in [0, 1].
            - ``'agg_data'``: DataFrame with bin means for ``x_col`` and ``y_col``.

    Raises:
        ValueError: If ``x_col`` or ``y_col`` are not columns in ``df``.

    Examples:
        >>> result = compute_aggregated_pearson(df, 'Packet_Loss_%', 'Error_Rate')
        >>> print(result['r'], result['p_value'])

    Requirements Addressed:
        - 17.1: Statistical analysis function
        - 17.6: Aggregated Pearson on binned data
        - 17.5: Handles fewer unique values than n_bins
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"compute_aggregated_pearson: column '{col}' not found in DataFrame.")

    work = df[[x_col, y_col]].dropna()

    if len(work) < n_bins:
        return {
            "r": float("nan"),
            "p_value": float("nan"),
            "agg_data": pd.DataFrame(columns=[x_col, y_col]),
        }

    try:
        work = work.copy()
        work["_bin"] = pd.qcut(work[x_col], q=n_bins, duplicates="drop")
    except ValueError:
        # Fewer unique values than bins — reduce bins
        n_unique = work[x_col].nunique()
        if n_unique < 2:
            return {
                "r": float("nan"),
                "p_value": float("nan"),
                "agg_data": pd.DataFrame(columns=[x_col, y_col]),
            }
        work = work.copy()
        work["_bin"] = pd.qcut(work[x_col], q=n_unique, duplicates="drop")

    agg = (
        work.groupby("_bin", observed=True)
        .agg(mean_x=(x_col, "mean"), mean_y=(y_col, "mean"))
        .dropna()
        .reset_index(drop=True)
    )
    agg.columns = [x_col, y_col]

    if len(agg) < 2:
        return {
            "r": float("nan"),
            "p_value": float("nan"),
            "agg_data": agg,
        }

    r, p_value = stats.pearsonr(agg[x_col], agg[y_col])
    return {
        "r": float(r),
        "p_value": float(p_value),
        "agg_data": agg,
    }


def compute_class_distribution(df: pd.DataFrame, column: str = "Efficiency_Status") -> Dict[str, float]:
    """Compute percentage distribution of a categorical column.

    Returns the percentage of rows belonging to each category.  The percentages
    always sum to 100% (within floating-point tolerance).

    Args:
        df: DataFrame containing the categorical column.
        column: Name of the categorical column.  Defaults to ``'Efficiency_Status'``.

    Returns:
        Dict mapping each unique category to its percentage (float in [0, 100]).
        Example: ``{'Low': 77.8, 'Medium': 11.2, 'High': 11.0}``

    Raises:
        ValueError: If ``column`` is not in ``df``.

    Requirements Addressed:
        - 21.1: Class distribution percentages sum to 100%
    """
    if column not in df.columns:
        raise ValueError(f"compute_class_distribution: column '{column}' not found in DataFrame.")

    total = len(df)
    if total == 0:
        return {}

    counts = df[column].value_counts()
    return {str(k): float(v / total * 100) for k, v in counts.items()}
