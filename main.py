import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import pandas as pd

tickers = ['AMD', 'QCOM']

# Run cointegration test
results = ct.coint_test(tickers)
data = results["data"]

# Print ADF Test Results
print("\n=== ADF Test Results ===")
for ticker, result in results["adf_results"].items():
    print(f"\nADF Test for {ticker}:")
    print(f"  Test Statistic : {result['Test Statistic']:.4f}")
    print(f"  p-value        : {result['p-value']:.4f}")
    print(f"  # Lags Used    : {result['# Lags Used']}")
    print("  Critical Values:")
    for key, value in result["Critical Values"].items():
        print(f"    {key}: {value:.4f}")

# Print Engle-Granger Cointegration Test
print("\n=== Engle-Granger Cointegration Test ===")
print(f"  Test Statistic: {results['coint_test']['Test Statistic']:.4f}")
print(f"  p-value       : {results['coint_test']['p-value']:.4f}")

# Apply Kalman Filter for Dynamic Hedge Ratio Estimation
kf_model = kf.KalmanFilterReg()
hedge_ratios = kf_model.run_kalman_filter(data[tickers[1]], data[tickers[0]])

# Create a DataFrame for Hedge Ratios
hedge_ratios_df = pd.DataFrame({
    "Date": data.index,
    "Hedge Ratio": hedge_ratios
}).set_index("Date")

# Print Hedge Ratios DataFrame
print("\n=== Dynamic Hedge Ratios (First 10 Values) ===")
print(hedge_ratios_df.head(10))

# Calculate the dynamic spread
spread_vec = data[tickers[0]] - hedge_ratios * data[tickers[1]]

# Generate Trading Signals
signals_df, spread_mean, spread_std = sg.generate_signals(spread_vec)

# Plot the combined visualization
sg.plot_combined_chart(data, tickers, spread_vec, signals_df)
