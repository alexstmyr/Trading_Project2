import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import pandas as pd

tickers = ['MSFT', 'AMD']

# Run cointegration test
results = ct.coint_test(tickers)

# Print ADF Test Results
for ticker, result in results["adf_results"].items():
    print(f"ADF Test for {ticker}:")
    print(f"  Test Statistic : {result['Test Statistic']:.4f}")
    print(f"  p-value        : {result['p-value']:.4f}")
    print("------")

# Print Engle-Granger Cointegration Test
print("\nEngle-Granger Cointegration Test:")
print(f"  Test Statistic: {results['coint_test']['Test Statistic']:.4f}")
print(f"  p-value       : {results['coint_test']['p-value']:.4f}")

# Print Johansen Cointegration Coefficients
beta_x, beta_y = results["johansen_beta"]
print(f"\nCointegration Spread Model: u_t = {beta_x:.5f} * x_t + {beta_y:.5f} * y_t")

# Print Hedge Ratio from OLS
print(f"\nHedge Ratio (OLS Regression): {results['hedge_ratio']:.4f}")

# Apply Kalman Filter for Dynamic Hedge Ratio Estimation
data = ct.download_data(tickers)
kf_model = kf.KalmanFilterReg()
hedge_ratios, kalman_preds = kf_model.run_kalman_filter(data[tickers[1]], data[tickers[0]])

# Convert to DataFrame for easier visualization
kalman_df = pd.DataFrame({
    "Date": data.index,
    "Hedge Ratio": hedge_ratios,
    "Predicted Spread": kalman_preds
}).set_index("Date")

# Print Kalman Filter results
print("\nKalman Filter Hedge Ratio (First 10 Values):")
print(kalman_df.head(10))

# Generate Trading Signals
signals_df, spread_mean, spread_std = sg.generate_signals(kalman_df["Predicted Spread"])

# Plot historical prices
sg.plot_prices(data, tickers)

# Plot spread and trading signals
sg.plot_spread_and_signals(kalman_df["Predicted Spread"], signals_df, spread_mean, spread_std)
