# Statistical Analysis of 6G Network Performance Impact on Smart Manufacturing Efficiency

**Date**: August 2026  
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
Performance Indicators are computed: the Network Stability Index (NSI=0.450),
Latency Sensitivity Score (LSS), Packet Loss Impact Ratio (PLIR), and the
Network-Efficiency Correlation via chi-square test and Cramér's V (V=0.006).

Findings are reported transparently regardless of statistical significance. The dataset
exhibits characteristics consistent with synthetic or simulated data (clean ranges,
no missing values), which limits the generalizability of findings to real-world deployments.

---

## 1. Introduction

The integration of 6G Ultra-Reliable Low-Latency Communication (URLLC) technology into
smart manufacturing environments promises to enable real-time control, precise coordination,
and data-driven decision-making. However, the practical impact of 6G network performance
on manufacturing outcomes remains an active area of research.

**Research Question**: Is there a statistically significant and practically meaningful
relationship between 6G network performance metrics (latency, packet loss) and
manufacturing efficiency outcomes (efficiency status, error rates, quality defect rates)?

**Dataset Characteristics**:
- **Rows**: 100,000
- **Machines**: 50
- **Time Range**: 2025-01-01 to 2025-03-10 (~69 days)
- **Date Format**: DD-MM-YYYY (explicitly parsed to prevent day/month ambiguity)
- **Columns**: 14 (plus 1 derived column: Network_Quality_Band)

---

## 2. Methodology

### 2.1 Data Preprocessing

The dataset was loaded and validated against the following schema requirements:
- Exactly 100,000 rows and 14 columns
- Required columns present: `Network_Latency_ms`, `Packet_Loss_%`, `Efficiency_Status`,
  `Production_Speed`, `Error_Rate`, `Quality_Control_Defect_Rate`, `Operation_Mode`,
  `Timestamp`, `Machine_ID`
- `Timestamp` parsed using explicit DD-MM-YYYY format: `pd.to_datetime(format='%d-%m-%Y')`
- `Network_Latency_ms` ≥ 0; `Packet_Loss_%` ∈ [0, 100]
- `Efficiency_Status` ∈ {Low, Medium, High}; `Operation_Mode` ∈ {Active, Idle, Maintenance}

**Efficiency Status Distribution**: High: 7.1%, Low: 77.7%, Medium: 15.2%

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

$$\text{NSI} = 1 - \frac{\text{CV}_{\text{latency}} + \text{CV}_{\text{packet\_loss}}}2$$

where $\text{CV} = \min\left(\frac{\sigma}{\mu}, 1\right)$ is the coefficient of variation,
capped at 1.0. NSI ∈ [0, 1], where 1 = perfectly stable.

#### KPI 2: Latency Sensitivity Score (LSS)

Quantifies how manufacturing efficiency changes with network latency using an aggregated
regression approach (row-level correlations ≈ 0).

Method: Encode `Efficiency_Status` as ordinal (Low=1, Medium=2, High=3), bin latency into
10 deciles, compute mean efficiency score per bin, fit linear regression:

$$\bar{E}_i = \beta_0 + \beta_1 \cdot \bar{L}_i + \epsilon_i, \quad \text{LSS} = |\hat{\beta}_1|$$

#### KPI 3: Packet Loss Impact Ratio (PLIR)

Compares mean error rates between high packet-loss (top quartile, Q4) and low packet-loss
(bottom quartile, Q1) groups using Welch's t-test:

$$\text{PLIR} = \frac{\bar{E}_{\text{Q4}}}{\bar{E}_{\text{Q1}}}$$

#### KPI 4: Network-Efficiency Correlation (NEC)

Tests statistical association between network quality band and efficiency class using
a chi-square test of independence and Cramér's V effect size:

$$V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, c) - 1)}}$$

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

## 3. Results

### 3.1 Network Stability Index

| Metric | Value |
|--------|-------|
| NSI | **0.450** |
| CV Latency | 0.521 |
| CV Packet Loss | 0.578 |

*Interpretation*: Network performance shows notable instability (NSI=0.450). High variability in latency or packet loss may impact manufacturing reliability.

### 3.2 Latency Sensitivity Score

| Metric | Value |
|--------|-------|
| LSS (|slope|) | **0.0001** |
| R² | 0.2711 |
| p-value | 0.1228 |

*Interpretation*: Minimal latency sensitivity observed (LSS=0.0001, p=0.1228). Efficiency appears not statistically significantly related to latency at the aggregated level.

### 3.3 Packet Loss Impact Ratio

| Metric | Value |
|--------|-------|
| PLIR | **1.000** |
| Q1 Mean Error Rate | 0.0253 |
| Q4 Mean Error Rate | 0.0253 |
| Welch's t-test p-value | 0.9381 |

*Interpretation*: Slight increase in error rates with high packet loss (PLIR=1.00, p=0.9381). The difference is not statistically significant.

### 3.4 Network-Efficiency Correlation (Chi-square + Cramér's V)

| Metric | Value |
|--------|-------|
| Cramér's V | **0.006** |
| χ² statistic | 6.78 |
| p-value | 0.1479 |

*Interpretation*: A weak not statistically significant association between network quality and efficiency was found (Cramér's V=0.006, p=0.1479).

**Contingency Table (Network_Quality_Band × Efficiency_Status)**:

| Network_Quality_Band | High | Low | Medium |
|---|---|---|---|
| High | 104 | 1220 | 227 |
| Low | 5375 | 59322 | 11480 |
| Medium | 1590 | 17183 | 3499 |



### 3.5 Visualizations

**Figure 1 — Efficiency Class Distribution by Network Quality Band:**

![Efficiency Distribution](figures\efficiency_distribution.png)

**Figure 2 — Network Latency Distribution:**

![Latency Distribution](figures\latency_distribution.png)

**Figure 3 — Low Efficiency Rate Risk Heatmap:**

![Risk Heatmap](figures\risk_heatmap.png)

---

## 4. Discussion

The analysis does not reveal strong, statistically significant associations between 6G network performance metrics and manufacturing efficiency outcomes at conventional significance levels. This result is reported transparently and may reflect characteristics of the dataset rather than the absence of any real-world effect.

**Key Findings**:
1. **Network Stability (NSI=0.450)**: Network performance shows notable instability (NSI=0.450). High variability in latency or packet loss may impact manufacturing reliability.
2. **Latency Sensitivity (LSS=0.0001)**: Minimal latency sensitivity observed (LSS=0.0001, p=0.1228). Efficiency appears not statistically significantly related to latency at the aggregated level.
3. **Packet Loss Impact (PLIR=1.000)**: Slight increase in error rates with high packet loss (PLIR=1.00, p=0.9381). The difference is not statistically significant.
4. **Network-Efficiency Association (V=0.006)**: A weak not statistically significant association between network quality and efficiency was found (Cramér's V=0.006, p=0.1479).

**Class Imbalance**: The dataset shows 77.7% of readings classified as Low efficiency.
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

3. **Class imbalance**: 77.7% Low efficiency prevalence reduces statistical power
   for detecting efficiency-related effects.

4. **Confounding factors**: Machine type, production batch, operator experience, and other
   unmeasured variables may influence both network conditions and manufacturing efficiency.

5. **Temporal granularity**: Daily-level aggregation used for trend analysis; intra-day
   patterns are not captured.

---

## 7. Conclusion

This study analyzed the statistical relationship between 6G network performance and smart
manufacturing efficiency using 100,000 telemetry records from 50 machines.

The analysis does not reveal strong, statistically significant associations between 6G network performance metrics and manufacturing efficiency outcomes at conventional significance levels. This result is reported transparently and may reflect characteristics of the dataset rather than the absence of any real-world effect.

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
*Report generated: 2026-08-27 08:21:17*
