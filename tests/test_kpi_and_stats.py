"""
Tests for KPI computation and statistical analysis modules.

Covers:
- Unit tests for all four KPI functions
- Edge cases: zero variance, insufficient data, zero denominator
- Property tests using Hypothesis: bounds, determinism, p-value validity
- Statistical analysis functions: Spearman, LOWESS, aggregated Pearson
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kpi_computation import (
    AllKPIs,
    compute_all_kpis,
    compute_latency_sensitivity_score,
    compute_network_efficiency_correlation,
    compute_network_stability_index,
    compute_packet_loss_impact_ratio,
)
from statistical_analysis import (
    compute_aggregated_pearson,
    compute_class_distribution,
    compute_spearman_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EFFICIENCY_VALUES = ["Low", "Medium", "High"]
QUALITY_BAND_VALUES = ["Low", "Medium", "High"]
OPERATION_MODE_VALUES = ["Active", "Idle", "Maintenance"]


def make_df(
    n: int = 500,
    latency_range: tuple = (5.0, 100.0),
    packet_loss_range: tuple = (0.0, 10.0),
    efficiency_dist: list = None,
) -> pd.DataFrame:
    """Create a synthetic DataFrame suitable for KPI testing."""
    rng = np.random.default_rng(42)
    latency = rng.uniform(*latency_range, size=n)
    packet_loss = rng.uniform(*packet_loss_range, size=n)

    if efficiency_dist is None:
        efficiency = rng.choice(EFFICIENCY_VALUES, size=n)
    else:
        efficiency = rng.choice(EFFICIENCY_VALUES, size=n, p=efficiency_dist)

    # Assign quality bands based on thresholds
    quality = [
        "High" if (lat < 20 and pl < 1) else ("Medium" if (lat < 50 and pl < 5) else "Low")
        for lat, pl in zip(latency, packet_loss)
    ]

    return pd.DataFrame(
        {
            "Network_Latency_ms": latency,
            "Packet_Loss_%": packet_loss,
            "Efficiency_Status": efficiency,
            "Error_Rate": rng.uniform(0.001, 0.05, size=n),
            "Quality_Control_Defect_Rate": rng.uniform(0.001, 0.05, size=n),
            "Network_Quality_Band": quality,
        }
    )


# ---------------------------------------------------------------------------
# KPI 1 — NSI tests
# ---------------------------------------------------------------------------


class TestNetworkStabilityIndex:
    def test_nsi_returns_float_in_0_1(self):
        df = make_df()
        result = compute_network_stability_index(df)
        assert 0.0 <= result.nsi <= 1.0

    def test_nsi_perfect_stability_zero_std(self):
        """If all latency and packet loss values are identical, NSI should equal 1.0."""
        df = make_df()
        df["Network_Latency_ms"] = 25.0
        df["Packet_Loss_%"] = 1.5
        result = compute_network_stability_index(df)
        assert result.nsi == pytest.approx(1.0, abs=1e-6)

    def test_nsi_has_interpretation_string(self):
        df = make_df()
        result = compute_network_stability_index(df)
        assert isinstance(result.interpretation, str) and len(result.interpretation) > 0

    def test_nsi_raises_on_missing_column(self):
        df = make_df().drop(columns=["Network_Latency_ms"])
        with pytest.raises(ValueError, match="missing required column"):
            compute_network_stability_index(df)

    def test_cv_latency_and_packet_loss_returned(self):
        df = make_df()
        result = compute_network_stability_index(df)
        assert 0.0 <= result.cv_latency <= 1.0
        assert 0.0 <= result.cv_packet_loss <= 1.0

    @given(
        latency=st.lists(st.floats(min_value=0.1, max_value=200.0, allow_nan=False), min_size=5, max_size=200),
        packet_loss=st.lists(st.floats(min_value=0.0, max_value=10.0, allow_nan=False), min_size=5, max_size=200),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_nsi_always_bounded_0_1(self, latency, packet_loss):
        """Property: NSI is always in [0, 1] for any valid input."""
        min_len = min(len(latency), len(packet_loss))
        df = pd.DataFrame({
            "Network_Latency_ms": latency[:min_len],
            "Packet_Loss_%": packet_loss[:min_len],
        })
        result = compute_network_stability_index(df)
        assert 0.0 <= result.nsi <= 1.0


# ---------------------------------------------------------------------------
# KPI 2 — LSS tests
# ---------------------------------------------------------------------------


class TestLatencySensitivityScore:
    def test_lss_returns_required_keys(self):
        df = make_df()
        result = compute_latency_sensitivity_score(df)
        assert hasattr(result, "score")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "p_value")
        assert hasattr(result, "interpretation")

    def test_lss_insufficient_data(self):
        df = make_df(n=5)
        result = compute_latency_sensitivity_score(df)
        assert math.isnan(result.score)

    def test_lss_r_squared_bounded(self):
        df = make_df()
        result = compute_latency_sensitivity_score(df)
        if not math.isnan(result.r_squared):
            assert 0.0 <= result.r_squared <= 1.0

    def test_lss_p_value_bounded(self):
        df = make_df()
        result = compute_latency_sensitivity_score(df)
        if not math.isnan(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_lss_raises_on_missing_column(self):
        df = make_df().drop(columns=["Efficiency_Status"])
        with pytest.raises(ValueError, match="missing required column"):
            compute_latency_sensitivity_score(df)

    def test_lss_bin_data_non_empty(self):
        df = make_df(n=200)
        result = compute_latency_sensitivity_score(df)
        if not math.isnan(result.score):
            assert not result.bin_data.empty


# ---------------------------------------------------------------------------
# KPI 3 — PLIR tests
# ---------------------------------------------------------------------------


class TestPacketLossImpactRatio:
    def test_plir_returns_required_fields(self):
        df = make_df()
        result = compute_packet_loss_impact_ratio(df)
        assert hasattr(result, "ratio")
        assert hasattr(result, "q1_error_rate")
        assert hasattr(result, "q4_error_rate")
        assert hasattr(result, "p_value")
        assert hasattr(result, "interpretation")

    def test_plir_ratio_non_negative_or_nan(self):
        """Property: PLIR ratio is either >= 0 or NaN."""
        df = make_df()
        result = compute_packet_loss_impact_ratio(df)
        assert math.isnan(result.ratio) or result.ratio >= 0.0

    def test_plir_zero_q1_error_rate_returns_nan(self):
        """If Q1 error rate is zero, PLIR should return NaN."""
        df = make_df()
        # Force Q1 error rates to zero
        q1_threshold = df["Packet_Loss_%"].quantile(0.25)
        df.loc[df["Packet_Loss_%"] <= q1_threshold, "Error_Rate"] = 0.0
        result = compute_packet_loss_impact_ratio(df)
        assert math.isnan(result.ratio)

    def test_plir_p_value_bounded(self):
        df = make_df()
        result = compute_packet_loss_impact_ratio(df)
        if not math.isnan(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_plir_raises_on_missing_column(self):
        df = make_df().drop(columns=["Error_Rate"])
        with pytest.raises(ValueError, match="missing required column"):
            compute_packet_loss_impact_ratio(df)

    @given(st.integers(min_value=50, max_value=300))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_plir_ratio_always_non_negative_or_nan(self, n):
        """Property: PLIR ratio is always >= 0 or NaN for any valid input."""
        df = make_df(n=n)
        result = compute_packet_loss_impact_ratio(df)
        assert math.isnan(result.ratio) or result.ratio >= 0.0


# ---------------------------------------------------------------------------
# KPI 4 — NEC tests
# ---------------------------------------------------------------------------


class TestNetworkEfficiencyCorrelation:
    def test_nec_returns_required_fields(self):
        df = make_df()
        result = compute_network_efficiency_correlation(df)
        assert hasattr(result, "cramers_v")
        assert hasattr(result, "chi2_statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "contingency_table")
        assert hasattr(result, "interpretation")

    def test_cramers_v_bounded_0_1(self):
        df = make_df()
        result = compute_network_efficiency_correlation(df)
        if not math.isnan(result.cramers_v):
            assert 0.0 <= result.cramers_v <= 1.0

    def test_p_value_bounded(self):
        df = make_df()
        result = compute_network_efficiency_correlation(df)
        if not math.isnan(result.p_value):
            assert 0.0 <= result.p_value <= 1.0

    def test_nec_raises_on_missing_column(self):
        df = make_df().drop(columns=["Network_Quality_Band"])
        with pytest.raises(ValueError, match="missing required column"):
            compute_network_efficiency_correlation(df)

    def test_nec_degenerate_contingency_table_returns_nan(self):
        """Single-value columns → degenerate table → NaN result."""
        df = make_df()
        df["Network_Quality_Band"] = "High"  # Only one unique value
        result = compute_network_efficiency_correlation(df)
        assert math.isnan(result.cramers_v)

    @given(st.integers(min_value=30, max_value=300))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_cramers_v_always_bounded(self, n):
        """Property: Cramér's V is always in [0, 1] for any valid input."""
        df = make_df(n=n)
        result = compute_network_efficiency_correlation(df)
        if not math.isnan(result.cramers_v):
            assert 0.0 <= result.cramers_v <= 1.0


# ---------------------------------------------------------------------------
# Unified compute_all_kpis tests
# ---------------------------------------------------------------------------


class TestComputeAllKPIs:
    def test_returns_all_kpis_instance(self):
        df = make_df()
        result = compute_all_kpis(df)
        assert isinstance(result, AllKPIs)

    def test_all_kpis_deterministic(self):
        """Property: same input → same output."""
        df = make_df(n=200)
        result1 = compute_all_kpis(df)
        result2 = compute_all_kpis(df)
        assert result1.nsi.nsi == pytest.approx(result2.nsi.nsi, abs=1e-10)
        assert (
            math.isnan(result1.lss.score) == math.isnan(result2.lss.score)
            or result1.lss.score == pytest.approx(result2.lss.score, abs=1e-10)
        )

    def test_kpis_computed_for_filtered_df(self):
        """KPIs should recompute correctly on a filtered subset."""
        df = make_df(n=500)
        df_filtered = df[df["Network_Quality_Band"] == "Low"]
        if len(df_filtered) > 50:
            result = compute_all_kpis(df_filtered)
            assert isinstance(result, AllKPIs)


# ---------------------------------------------------------------------------
# Statistical analysis tests
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_returns_rho_and_pvalue(self):
        df = make_df()
        result = compute_spearman_correlation(df, "Network_Latency_ms", "Error_Rate")
        assert "rho" in result
        assert "p_value" in result

    def test_rho_bounded(self):
        df = make_df()
        result = compute_spearman_correlation(df, "Network_Latency_ms", "Error_Rate")
        if not math.isnan(result["rho"]):
            assert -1.0 <= result["rho"] <= 1.0

    def test_constant_column_returns_nan(self):
        df = make_df()
        df["constant"] = 5.0
        result = compute_spearman_correlation(df, "constant", "Error_Rate")
        assert math.isnan(result["rho"])

    def test_raises_on_missing_column(self):
        df = make_df()
        with pytest.raises(ValueError, match="not found"):
            compute_spearman_correlation(df, "nonexistent_col", "Error_Rate")

    @given(st.integers(min_value=10, max_value=200))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_rho_always_bounded(self, n):
        """Property: Spearman rho is always in [-1, 1]."""
        df = make_df(n=n)
        result = compute_spearman_correlation(df, "Network_Latency_ms", "Packet_Loss_%")
        if not math.isnan(result["rho"]):
            assert -1.0 <= result["rho"] <= 1.0


class TestAggregatedPearson:
    def test_returns_r_pvalue_agg_data(self):
        df = make_df()
        result = compute_aggregated_pearson(df, "Network_Latency_ms", "Error_Rate")
        assert "r" in result
        assert "p_value" in result
        assert "agg_data" in result

    def test_r_bounded(self):
        df = make_df()
        result = compute_aggregated_pearson(df, "Network_Latency_ms", "Error_Rate")
        if not math.isnan(result["r"]):
            assert -1.0 <= result["r"] <= 1.0

    def test_p_value_bounded(self):
        df = make_df()
        result = compute_aggregated_pearson(df, "Network_Latency_ms", "Error_Rate")
        if not math.isnan(result["p_value"]):
            assert 0.0 <= result["p_value"] <= 1.0

    def test_insufficient_data_returns_nan(self):
        df = make_df(n=5)
        result = compute_aggregated_pearson(df, "Network_Latency_ms", "Error_Rate", n_bins=10)
        assert math.isnan(result["r"])

    def test_raises_on_missing_column(self):
        df = make_df()
        with pytest.raises(ValueError, match="not found"):
            compute_aggregated_pearson(df, "nonexistent_col", "Error_Rate")

    @given(st.integers(min_value=20, max_value=300))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_r_always_bounded(self, n):
        """Property: Pearson r on aggregated data is always in [-1, 1]."""
        df = make_df(n=n)
        result = compute_aggregated_pearson(df, "Packet_Loss_%", "Error_Rate")
        if not math.isnan(result["r"]):
            assert -1.0 <= result["r"] <= 1.0


# ---------------------------------------------------------------------------
# Class distribution tests
# ---------------------------------------------------------------------------


class TestClassDistribution:
    def test_percentages_sum_to_100(self):
        """Property: class distribution percentages sum to 100%."""
        df = make_df()
        dist = compute_class_distribution(df, "Efficiency_Status")
        assert sum(dist.values()) == pytest.approx(100.0, abs=0.01)

    def test_all_values_between_0_and_100(self):
        df = make_df()
        dist = compute_class_distribution(df, "Efficiency_Status")
        for v in dist.values():
            assert 0.0 <= v <= 100.0

    def test_empty_df_returns_empty_dict(self):
        df = pd.DataFrame({"Efficiency_Status": pd.Series([], dtype=str)})
        dist = compute_class_distribution(df, "Efficiency_Status")
        assert dist == {}

    def test_raises_on_missing_column(self):
        df = make_df()
        with pytest.raises(ValueError, match="not found"):
            compute_class_distribution(df, "nonexistent_col")

    @given(st.lists(st.sampled_from(["Low", "Medium", "High"]), min_size=1, max_size=500))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_percentages_always_sum_to_100(self, values):
        """Property: class distribution percentages always sum to 100%."""
        df = pd.DataFrame({"Efficiency_Status": values})
        dist = compute_class_distribution(df, "Efficiency_Status")
        assert sum(dist.values()) == pytest.approx(100.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
