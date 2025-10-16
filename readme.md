determine market regime/trendingness
use to identify if going long or short, bull or bear spread


examine changes in option contract ivol over time

greeks attribution
IV Rank:
\frac{\text{Current IV} - \text{Min IV (N days)}}{\text{Max IV (N days)} - \text{Min IV (N days)}}
→ positions current IV within its lookback range (0–100%).
IV Percentile:
% of days in the lookback where IV ≤ current IV.
IV vs Realized Volatility (RV):
Compare IV to historical realized vol → identifies if options are “rich” (IV > RV) or “cheap” (IV < RV).
Rolling averages / volatility smoothing
20-day / 60-day IV averages to detect regime shifts.

Stationarity tests (ADF, KPSS)
Check if IV series mean-reverts (it usually does).

Cointegration tests
Compare IV of related assets (e.g., SPY vs QQQ) for relative value.

ARIMA / GARCH models
Forecast volatility dynamics and term structure.

Term Structure Slope: short-term IV vs long-term IV (steep, flat, inverted).
Skew / Smile Measures: OTM put IV – ATM IV, or call vs put skew.
Principal Component Analysis (PCA)
Decompose IV surface moves into level, slope, and curvature factors.

Vol of Vol (VVIX)
Analyze second-order volatility risk (how IV itself fluctuates).

Pre/Post Event IV Analysis
Track IV before and after earnings, CPI, FOMC.
Compute average “IV crush” magnitude.

Regression on macro variables
IV = f(VIX, MOVE index, credit spreads, yields, dollar).

Z-Score of IV vs History
Standardize IV relative to mean/σ to spot extremes.

Pairs Trading on IV
Long vol in one ETF/stock, short vol in another if historical spread widens.

Kalman Filter
Adaptive estimation of fair IV based on regime.

IV Forecast Accuracy:
Test how well IV predicted realized vol.
Greeks Attribution:
Break down daily option P/L into delta/theta vs vega (IV change).
Monte Carlo Simulation:
Use IV time series to simulate forward paths for hedging strategies.



If V = option value, then the total change in option value (\Delta V) over a short period can be approximated by a Taylor expansion:



\Delta V \approx \Delta \cdot \Delta S + \tfrac{1}{2}\Gamma \cdot (\Delta S)^2 + \Theta \cdot \Delta t + \nu \cdot \Delta \sigma + \rho \cdot \Delta r



Where:



\Delta S = change in underlying price
\Delta \sigma = change in implied vol
\Delta t = time passed (usually 1 day)
\Delta r = change in interest rate

