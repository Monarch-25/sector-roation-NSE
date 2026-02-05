# Custom Benchmark Logic

## Overview
Originally, the proposed benchmark for this sector rotation strategy was the Nifty 500 index. However, the 10 sector indices used in this analysis do not perfectly replicate the Nifty 500 due to differences in sector composition and weighting.

To ensure the benchmark is fully aligned with the investable universe of sector indices, we have implemented a **'Custom_Benchmark'**.

## Calculation Logic
The Custom_Benchmark is constructed using an equal-weighted combination of all sector indices included in the dataset:

1. **Daily Returns**: Calculate the daily percentage change (`pct_change()`) for each of the 10 sector indices.
2. **Mean Return**: For each trading day, calculate the average (mean) of these daily returns across all sectors.
3. **Price Series Conversion**: The mean daily returns are converted back into a cumulative price series (index), starting with an initial value of **1.0**.

## Included Sectors
The benchmark is an equal-weighted average of the following sectors:
- Auto
- Finance
- Oil & Gas (Energy)
- FMCG
- IT
- Metal
- Consumer
- Realty
- Media
- Healthcare (Pharma)

## Rationale
Using an equal-weighted average of the constituent sectors ensures that the benchmark represents the "average" performance of the specifically selected sectors, providing a more accurate baseline for evaluating the performance of the rotation strategy within this universe.
