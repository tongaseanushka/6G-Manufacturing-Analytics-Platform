"""
Streamlit Dashboard for Thales 6G Manufacturing Analytics Platform.

This is the main entry point for the interactive dashboard.  It provides four
analytical modules for exploring the relationship between 6G network performance
and smart manufacturing efficiency.

Usage:
    streamlit run src/dashboard.py

Modules:
    Module 1 — Network Performance Overview
        Time-series charts for daily latency/packet loss trends + 4 KPI scorecards.

    Module 2 — Network vs Efficiency
        Stacked bar chart for efficiency distribution by network quality band;
        scatter plot with row-level or aggregated binned view toggle.

    Module 3 — Quality & Error Impact
        Aggregated scatter plots for packet loss vs error rate and latency vs
        defect rate; error rate summary statistics by network quality band.

    Module 4 — 6G Optimization Insights
        Latency/packet-loss threshold recommendations, plain-language recommendations,
        and a 2D efficiency risk heatmap.

Performance:
    - Data loading: @st.cache_data (no TTL — static dataset)
    - KPI computation: @st.cache_data(ttl=3600, max_entries=50)
    - Target: initial load < 5s, filter updates < 3s

Requirements Addressed:
    - 7.x – 10.x: Dashboard modules
    - 11.x – 14.x: Global filters
    - 18.x: Performance and caching
    - 20.3: Docstrings and type hints
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is on the path regardless of where streamlit is launched from
sys.path.insert(0, str(Path(__file__).parent))

from data_prep import DataValidationError, load_and_prepare_dataset
from kpi_computation import AllKPIs, compute_all_kpis
from statistical_analysis import (
    compute_aggregated_pearson,
    compute_class_distribution,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Thales 6G Manufacturing Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dataset path resolution
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "data" / "Thales_Group_Manufacturing.csv"


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------


def initialize_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Build default filter state from the full dataset.

    Args:
        df: Full validated DataFrame.

    Returns:
        Dict with keys:
            - ``'quality_bands'``: All unique Network_Quality_Band values.
            - ``'efficiency_classes'``: All unique Efficiency_Status values.
            - ``'operation_modes'``: All unique Operation_Mode values.
            - ``'date_start'``: Minimum Timestamp date.
            - ``'date_end'``: Maximum Timestamp date.

    Requirements Addressed:
        - 11.1, 12.1, 13.1, 14.1: Default filter values
    """
    return {
        "quality_bands": sorted(df["Network_Quality_Band"].unique().tolist()),
        "efficiency_classes": sorted(df["Efficiency_Status"].unique().tolist()),
        "operation_modes": sorted(df["Operation_Mode"].unique().tolist()),
        "date_start": df["Timestamp"].min().date(),
        "date_end": df["Timestamp"].max().date(),
    }


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply global sidebar filters to the DataFrame.

    Filtering is always monotonic: the result is a subset of the input DataFrame
    (never larger).

    Args:
        df: Full validated DataFrame.
        filters: Dict produced by the sidebar controls.

    Returns:
        Filtered DataFrame.

    Side Effects:
        Calls ``st.error`` / ``st.warning`` and ``st.stop()`` for invalid states.

    Requirements Addressed:
        - 11.3, 12.3, 13.3, 14.3: Filter application
        - 11.5, 14.4: Date validation
    """
    start = filters["date_start"]
    end = filters["date_end"]

    if start > end:
        st.error("⚠️ Start date must be on or before end date.")
        st.stop()

    mask = (
        df["Network_Quality_Band"].isin(filters["quality_bands"])
        & df["Efficiency_Status"].isin(filters["efficiency_classes"])
        & df["Operation_Mode"].isin(filters["operation_modes"])
        & (df["Timestamp"].dt.date >= start)
        & (df["Timestamp"].dt.date <= end)
    )
    filtered = df.loc[mask]

    if filtered.empty:
        st.warning("⚠️ No data matches the selected filters. Please broaden your selection.")
        st.stop()

    return filtered


# ---------------------------------------------------------------------------
# Module 1 — Network Performance Overview
# ---------------------------------------------------------------------------


def render_network_performance_overview(
    df_filtered: pd.DataFrame, kpis: AllKPIs
) -> None:
    """Render Module 1: Network Performance Overview.

    Displays:
    - Daily average latency trend (line chart)
    - Daily average packet loss trend (line chart)
    - KPI scorecard row (NSI, LSS, PLIR, NEC)

    Args:
        df_filtered: Filtered DataFrame.
        kpis: Pre-computed AllKPIs result.

    Requirements Addressed:
        - 7.1–7.7: Network performance time-series and KPI scorecards
    """
    st.subheader("📡 Module 1 — Network Performance Overview")

    # Daily aggregation
    daily = (
        df_filtered.set_index("Timestamp")
        .resample("D")[["Network_Latency_ms", "Packet_Loss_%"]]
        .mean()
        .reset_index()
    )

    col_lat, col_pl = st.columns(2)

    with col_lat:
        fig_lat = px.line(
            daily,
            x="Timestamp",
            y="Network_Latency_ms",
            title="Daily Average Network Latency",
            labels={"Network_Latency_ms": "Latency (ms)", "Timestamp": "Date"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_lat.update_layout(height=320)
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_pl:
        fig_pl = px.line(
            daily,
            x="Timestamp",
            y="Packet_Loss_%",
            title="Daily Average Packet Loss",
            labels={"Packet_Loss_%": "Packet Loss (%)", "Timestamp": "Date"},
            color_discrete_sequence=["#d62728"],
        )
        fig_pl.update_layout(height=320)
        st.plotly_chart(fig_pl, use_container_width=True)

    # KPI scorecards
    st.markdown("#### Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)

    nsi_val = kpis.nsi.nsi
    lss_val = kpis.lss.score
    plir_val = kpis.plir.ratio
    nec_val = kpis.nec.cramers_v

    with k1:
        st.metric(
            "Network Stability Index",
            f"{nsi_val:.3f}" if not np.isnan(nsi_val) else "N/A",
            help=f"0 = unstable, 1 = stable. {kpis.nsi.interpretation}",
        )
    with k2:
        st.metric(
            "Latency Sensitivity Score",
            f"{lss_val:.4f}" if not np.isnan(lss_val) else "N/A",
            help=kpis.lss.interpretation,
        )
    with k3:
        st.metric(
            "Packet Loss Impact Ratio",
            f"{plir_val:.2f}×" if not np.isnan(plir_val) else "N/A",
            help=kpis.plir.interpretation,
        )
    with k4:
        st.metric(
            "Network-Efficiency Correlation",
            f"V={nec_val:.3f}" if not np.isnan(nec_val) else "N/A",
            help=kpis.nec.interpretation,
        )


# ---------------------------------------------------------------------------
# Module 2 — Network vs Efficiency
# ---------------------------------------------------------------------------


def render_network_vs_efficiency(df_filtered: pd.DataFrame) -> None:
    """Render Module 2: Network vs Efficiency.

    Displays:
    - Stacked bar chart: efficiency class distribution by network quality band.
    - Scatter plot with toggle between row-level and aggregated binned views.

    Args:
        df_filtered: Filtered DataFrame.

    Requirements Addressed:
        - 8.1–8.6: Network vs efficiency visualizations
    """
    st.subheader("📊 Module 2 — Network vs Efficiency")

    # Stacked bar chart
    crosstab = (
        pd.crosstab(
            df_filtered["Network_Quality_Band"],
            df_filtered["Efficiency_Status"],
            normalize="index",
        )
        * 100
    )
    # Ensure consistent column order
    for col in ["Low", "Medium", "High"]:
        if col not in crosstab.columns:
            crosstab[col] = 0.0
    crosstab = crosstab[["Low", "Medium", "High"]].reset_index()

    crosstab_melted = crosstab.melt(
        id_vars="Network_Quality_Band",
        value_vars=["Low", "Medium", "High"],
        var_name="Efficiency_Status",
        value_name="Percentage",
    )

    fig_bar = px.bar(
        crosstab_melted,
        x="Network_Quality_Band",
        y="Percentage",
        color="Efficiency_Status",
        title="Efficiency Class Distribution by Network Quality Band",
        labels={"Percentage": "Percentage (%)", "Network_Quality_Band": "Network Quality Band"},
        color_discrete_map={"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"},
        barmode="stack",
        category_orders={"Network_Quality_Band": ["Low", "Medium", "High"]},
    )
    fig_bar.update_layout(height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter plot toggle
    view_mode = st.radio(
        "Scatter Plot View Mode",
        ["Row-Level", "Aggregated (Binned)"],
        horizontal=True,
    )

    col_a, col_b = st.columns(2)

    if view_mode == "Row-Level":
        work = df_filtered.copy()
        work["efficiency_score"] = work["Efficiency_Status"].map(
            {"Low": 1, "Medium": 2, "High": 3}
        )
        # Sample for performance
        sample = work.sample(min(5000, len(work)), random_state=42)

        with col_a:
            fig_a = px.scatter(
                sample,
                x="Network_Latency_ms",
                y="efficiency_score",
                title="Latency vs Efficiency (Row-Level)",
                labels={"Network_Latency_ms": "Latency (ms)", "efficiency_score": "Efficiency Score"},
                opacity=0.3,
                color_discrete_sequence=["#1f77b4"],
            )
            fig_a.update_layout(height=360)
            st.plotly_chart(fig_a, use_container_width=True)

        with col_b:
            fig_b = px.scatter(
                sample,
                x="Packet_Loss_%",
                y="efficiency_score",
                title="Packet Loss vs Efficiency (Row-Level)",
                labels={"Packet_Loss_%": "Packet Loss (%)", "efficiency_score": "Efficiency Score"},
                opacity=0.3,
                color_discrete_sequence=["#d62728"],
            )
            fig_b.update_layout(height=360)
            st.plotly_chart(fig_b, use_container_width=True)

    else:  # Aggregated
        # Latency vs efficiency (aggregated)
        work = df_filtered.copy()
        work["efficiency_score"] = work["Efficiency_Status"].map(
            {"Low": 1, "Medium": 2, "High": 3}
        )
        result_lat = compute_aggregated_pearson(work, "Network_Latency_ms", "efficiency_score")
        result_pl = compute_aggregated_pearson(work, "Packet_Loss_%", "efficiency_score")

        with col_a:
            agg_lat = result_lat["agg_data"]
            r_lat = result_lat["r"]
            p_lat = result_lat["p_value"]
            if not agg_lat.empty:
                fig_c = px.scatter(
                    agg_lat,
                    x="Network_Latency_ms",
                    y="efficiency_score",
                    trendline="ols",
                    title=f"Latency vs Efficiency (Aggregated) | r={r_lat:.3f}, p={p_lat:.4f}",
                    labels={"Network_Latency_ms": "Mean Latency (ms)", "efficiency_score": "Mean Efficiency Score"},
                    color_discrete_sequence=["#1f77b4"],
                )
                fig_c.update_layout(height=360)
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Insufficient data for aggregated latency vs efficiency view.")

        with col_b:
            agg_pl = result_pl["agg_data"]
            r_pl = result_pl["r"]
            p_pl = result_pl["p_value"]
            if not agg_pl.empty:
                fig_d = px.scatter(
                    agg_pl,
                    x="Packet_Loss_%",
                    y="efficiency_score",
                    trendline="ols",
                    title=f"Packet Loss vs Efficiency (Aggregated) | r={r_pl:.3f}, p={p_pl:.4f}",
                    labels={"Packet_Loss_%": "Mean Packet Loss (%)", "efficiency_score": "Mean Efficiency Score"},
                    color_discrete_sequence=["#d62728"],
                )
                fig_d.update_layout(height=360)
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.info("Insufficient data for aggregated packet loss vs efficiency view.")


# ---------------------------------------------------------------------------
# Module 3 — Quality & Error Impact
# ---------------------------------------------------------------------------


def render_quality_error_impact(df_filtered: pd.DataFrame) -> None:
    """Render Module 3: Quality & Error Impact.

    Displays:
    - Aggregated scatter plot: packet loss vs error rate.
    - Aggregated scatter plot: latency vs quality control defect rate.
    - Summary statistics table by network quality band.

    Args:
        df_filtered: Filtered DataFrame.

    Requirements Addressed:
        - 9.1–9.6: Quality and error impact module
    """
    st.subheader("🔬 Module 3 — Quality & Error Impact")

    col_a, col_b = st.columns(2)

    result_pl_err = compute_aggregated_pearson(df_filtered, "Packet_Loss_%", "Error_Rate")
    result_lat_def = compute_aggregated_pearson(
        df_filtered, "Network_Latency_ms", "Quality_Control_Defect_Rate"
    )

    with col_a:
        agg = result_pl_err["agg_data"]
        r = result_pl_err["r"]
        p = result_pl_err["p_value"]
        r_str = f"r={r:.3f}" if not np.isnan(r) else "r=N/A"
        p_str = f"p={p:.4f}" if not np.isnan(p) else "p=N/A"
        if not agg.empty:
            fig_a = px.scatter(
                agg,
                x="Packet_Loss_%",
                y="Error_Rate",
                trendline="ols",
                title=f"Packet Loss vs Error Rate (Aggregated) | {r_str}, {p_str}",
                labels={"Packet_Loss_%": "Mean Packet Loss (%)", "Error_Rate": "Mean Error Rate"},
                color_discrete_sequence=["#d62728"],
            )
            fig_a.update_layout(height=360)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("Insufficient data for packet loss vs error rate view.")

    with col_b:
        agg2 = result_lat_def["agg_data"]
        r2 = result_lat_def["r"]
        p2 = result_lat_def["p_value"]
        r2_str = f"r={r2:.3f}" if not np.isnan(r2) else "r=N/A"
        p2_str = f"p={p2:.4f}" if not np.isnan(p2) else "p=N/A"
        if not agg2.empty:
            fig_b = px.scatter(
                agg2,
                x="Network_Latency_ms",
                y="Quality_Control_Defect_Rate",
                trendline="ols",
                title=f"Latency vs Defect Rate (Aggregated) | {r2_str}, {p2_str}",
                labels={
                    "Network_Latency_ms": "Mean Latency (ms)",
                    "Quality_Control_Defect_Rate": "Mean Defect Rate",
                },
                color_discrete_sequence=["#ff7f0e"],
            )
            fig_b.update_layout(height=360)
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Insufficient data for latency vs defect rate view.")

    # Summary statistics table
    st.markdown("#### Error Rate Summary by Network Quality Band")
    summary = (
        df_filtered.groupby("Network_Quality_Band", observed=True)[
            ["Error_Rate", "Quality_Control_Defect_Rate"]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.columns = [" ".join(c).strip() for c in summary.columns]
    st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Module 4 — 6G Optimization Insights
# ---------------------------------------------------------------------------


def generate_recommendations(kpis: AllKPIs, df: pd.DataFrame) -> List[str]:
    """Generate plain-language optimization recommendations from KPI results.

    Args:
        kpis: AllKPIs result from compute_all_kpis().
        df: Filtered DataFrame (used for class imbalance check).

    Returns:
        List of recommendation strings.

    Requirements Addressed:
        - 10.4, 10.5: Recommendation generation
    """
    recs: List[str] = []

    # NSI recommendation
    nsi = kpis.nsi.nsi
    if not np.isnan(nsi) and nsi < 0.70:
        recs.append(
            f"🔴 **Network Instability Detected** (NSI={nsi:.3f}): High variability in latency or "
            "packet loss observed. Investigate network infrastructure for congestion points or "
            "failing hardware."
        )

    # LSS recommendation
    lss = kpis.lss.score
    lss_p = kpis.lss.p_value
    if not np.isnan(lss) and lss > 0.01 and not np.isnan(lss_p) and lss_p < 0.05:
        recs.append(
            f"🟠 **Latency-Efficiency Impact** (LSS={lss:.4f}): Statistically significant "
            "relationship detected between network latency and manufacturing efficiency. "
            "Reducing latency below 20ms may improve efficiency outcomes."
        )

    # PLIR recommendation
    plir = kpis.plir.ratio
    if not np.isnan(plir) and plir > 1.2:
        recs.append(
            f"🟠 **Packet Loss Impact** (PLIR={plir:.2f}×): High packet loss is associated with "
            f"{plir:.2f}× higher error rates. Target packet loss below 1% for optimal manufacturing quality."
        )

    # NEC recommendation
    nec_v = kpis.nec.cramers_v
    nec_p = kpis.nec.p_value
    if not np.isnan(nec_v) and not np.isnan(nec_p) and nec_p < 0.05 and nec_v > 0.1:
        recs.append(
            f"🟡 **Network-Efficiency Association** (Cramér's V={nec_v:.3f}): A statistically "
            "significant association between network quality band and efficiency class was found. "
            "Maintaining High-quality network conditions may support better efficiency outcomes."
        )

    # Class imbalance caveat
    dist = compute_class_distribution(df, "Efficiency_Status")
    low_pct = dist.get("Low", 0.0)
    if low_pct > 70.0:
        recs.append(
            f"ℹ️ **Class Imbalance Note**: {low_pct:.1f}% of readings show Low efficiency. "
            "Results should be interpreted with caution — the dataset is heavily skewed toward "
            "Low efficiency states."
        )

    if not recs:
        recs.append(
            "✅ **No critical network issues detected**: KPI values are within acceptable ranges. "
            "No statistically significant network-efficiency relationships were identified at this "
            "significance level. Consider monitoring trends over time."
        )

    return recs


def render_optimization_insights(df_filtered: pd.DataFrame, kpis: AllKPIs) -> None:
    """Render Module 4: 6G Optimization Insights.

    Displays:
    - Latency and packet loss threshold recommendations (where Low efficiency > 50%).
    - Plain-language recommendation bullets.
    - 2D efficiency risk heatmap (latency bins × packet loss bins).

    Args:
        df_filtered: Filtered DataFrame.
        kpis: Pre-computed AllKPIs result.

    Requirements Addressed:
        - 10.1–10.6: Optimization insights module
    """
    st.subheader("⚙️ Module 4 — 6G Optimization Insights")

    # ---- Threshold recommendations ----
    st.markdown("#### Threshold Recommendations")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        # Latency buckets
        try:
            df_lat = df_filtered.copy()
            df_lat["lat_bin"] = pd.cut(df_lat["Network_Latency_ms"], bins=5)
            lat_risk = (
                df_lat.groupby("lat_bin", observed=True)["Efficiency_Status"]
                .apply(lambda s: (s == "Low").mean() * 100)
                .reset_index()
            )
            lat_risk.columns = ["Latency Bin", "Low Efficiency Rate (%)"]
            risky = lat_risk[lat_risk["Low Efficiency Rate (%)"] > 50.0]
            if not risky.empty:
                threshold_bin = risky.iloc[0]["Latency Bin"]
                st.metric(
                    "⚠️ Latency Risk Threshold",
                    str(threshold_bin),
                    help="First latency bin where Low efficiency exceeds 50%.",
                )
            else:
                st.metric("✅ Latency Risk Threshold", "Not exceeded", help="Low efficiency rate < 50% across all latency bins.")
        except Exception:
            st.info("Unable to compute latency threshold.")

    with col_t2:
        # Packet loss buckets
        try:
            df_pl = df_filtered.copy()
            df_pl["pl_bin"] = pd.cut(df_pl["Packet_Loss_%"], bins=5)
            pl_risk = (
                df_pl.groupby("pl_bin", observed=True)["Efficiency_Status"]
                .apply(lambda s: (s == "Low").mean() * 100)
                .reset_index()
            )
            pl_risk.columns = ["Packet Loss Bin", "Low Efficiency Rate (%)"]
            risky_pl = pl_risk[pl_risk["Low Efficiency Rate (%)"] > 50.0]
            if not risky_pl.empty:
                threshold_bin_pl = risky_pl.iloc[0]["Packet Loss Bin"]
                st.metric(
                    "⚠️ Packet Loss Risk Threshold",
                    str(threshold_bin_pl),
                    help="First packet loss bin where Low efficiency exceeds 50%.",
                )
            else:
                st.metric("✅ Packet Loss Risk Threshold", "Not exceeded", help="Low efficiency rate < 50% across all packet loss bins.")
        except Exception:
            st.info("Unable to compute packet loss threshold.")

    # ---- Recommendations ----
    st.markdown("#### Recommendations")
    recs = generate_recommendations(kpis, df_filtered)
    for i, rec in enumerate(recs, start=1):
        st.markdown(f"{i}. {rec}")

    # ---- Risk Heatmap ----
    st.markdown("#### Efficiency Risk Heatmap")
    try:
        df_hm = df_filtered.copy()
        df_hm["lat_q"] = pd.qcut(df_hm["Network_Latency_ms"], q=5, duplicates="drop", labels=False)
        df_hm["pl_q"] = pd.qcut(df_hm["Packet_Loss_%"], q=5, duplicates="drop", labels=False)

        heatmap_data = (
            df_hm.groupby(["lat_q", "pl_q"], observed=True)["Efficiency_Status"]
            .apply(lambda s: (s == "Low").mean() * 100)
            .reset_index()
        )
        heatmap_data.columns = ["Latency Level", "Packet Loss Level", "Low Efficiency Rate (%)"]

        pivot = heatmap_data.pivot(
            index="Latency Level", columns="Packet Loss Level", values="Low Efficiency Rate (%)"
        )

        lat_labels = ["Very Low", "Low", "Medium", "High", "Very High"][: pivot.shape[0]]
        pl_labels = ["Very Low", "Low", "Medium", "High", "Very High"][: pivot.shape[1]]

        fig_hm = px.imshow(
            pivot,
            labels=dict(x="Packet Loss Level", y="Latency Level", color="Low Efficiency Rate (%)"),
            x=pl_labels,
            y=lat_labels,
            color_continuous_scale="Reds",
            title="Low Efficiency Rate by Network Conditions (%)",
            zmin=0,
            zmax=100,
        )
        fig_hm.update_layout(height=420)
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception as e:
        st.info(f"Unable to render risk heatmap: {e}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    """Render the global filter sidebar and return the active filter state.

    Args:
        df: Full validated DataFrame.

    Returns:
        Dict of active filter values.

    Requirements Addressed:
        - 11.x, 12.x, 13.x, 14.x: Global filters
    """
    defaults = initialize_filters(df)

    st.sidebar.title("📡 Thales 6G Analytics")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Global Filters")

    quality_bands = st.sidebar.multiselect(
        "Network Quality Band",
        options=sorted(df["Network_Quality_Band"].unique()),
        default=defaults["quality_bands"],
    )

    efficiency_classes = st.sidebar.multiselect(
        "Efficiency Class",
        options=sorted(df["Efficiency_Status"].unique()),
        default=defaults["efficiency_classes"],
    )

    operation_modes = st.sidebar.multiselect(
        "Operation Mode",
        options=sorted(df["Operation_Mode"].unique()),
        default=defaults["operation_modes"],
    )

    date_range = st.sidebar.date_input(
        "Time Window",
        value=(defaults["date_start"], defaults["date_end"]),
        min_value=defaults["date_start"],
        max_value=defaults["date_end"],
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_start, date_end = date_range[0], date_range[1]
    else:
        date_start = date_end = date_range

    # If any multiselect is empty, revert to all
    if not quality_bands:
        quality_bands = defaults["quality_bands"]
    if not efficiency_classes:
        efficiency_classes = defaults["efficiency_classes"]
    if not operation_modes:
        operation_modes = defaults["operation_modes"]

    return {
        "quality_bands": quality_bands,
        "efficiency_classes": efficiency_classes,
        "operation_modes": operation_modes,
        "date_start": date_start,
        "date_end": date_end,
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit application entry point."""
    st.title("📡 Thales 6G Manufacturing Analytics Platform")
    st.markdown(
        "Analyzing the statistical relationship between 6G network performance "
        "and smart manufacturing efficiency across 50 machines."
    )

    # ---- Data loading ----
    csv_path = str(DEFAULT_CSV_PATH)

    if not Path(csv_path).exists():
        st.warning(
            f"Dataset not found at `{csv_path}`. "
            "Please upload the `Thales_Group_Manufacturing.csv` file below."
        )
        uploaded = st.file_uploader("Upload Dataset CSV", type=["csv"])
        if uploaded is None:
            st.stop()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            csv_path = tmp.name

    with st.spinner("Loading and validating dataset…"):
        try:
            df = load_and_prepare_dataset(csv_path)
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            st.stop()
        except DataValidationError as e:
            st.error(f"Data validation failed: {e}")
            st.stop()

    # ---- Sidebar filters ----
    filters = render_sidebar(df)

    # ---- Apply filters ----
    with st.spinner("Applying filters…"):
        df_filtered = apply_filters(df, filters)

    st.sidebar.markdown("---")
    st.sidebar.info(f"**{len(df_filtered):,}** rows selected ({len(df_filtered)/len(df)*100:.1f}%)")

    # ---- Compute KPIs ----
    with st.spinner("Computing KPIs…"):
        kpis = compute_all_kpis(df_filtered)

    # ---- Dashboard tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Module 1 — Network Overview",
        "Module 2 — Network vs Efficiency",
        "Module 3 — Quality & Error Impact",
        "Module 4 — Optimization Insights",
    ])

    with tab1:
        render_network_performance_overview(df_filtered, kpis)

    with tab2:
        render_network_vs_efficiency(df_filtered)

    with tab3:
        render_quality_error_impact(df_filtered)

    with tab4:
        render_optimization_insights(df_filtered, kpis)

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "Thales 6G Manufacturing Analytics Platform · "
        "All KPI formulas documented in `src/kpi_computation.py` · "
        "Research paper: `python src/report_gen.py --output reports/research_paper.md`"
    )


if __name__ == "__main__":
    main()
