import cointegration_test as ct
import signals as sg

tickers = ['SHEL', 'CVX']

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

# Print Johansen Cointegration Vector
print(f"\nCointegrating vector (normalized): {results['johansen_beta']}")

# Generate Trading Signals
signals_df, spread_mean, spread_std = sg.generate_signals(results['spread_vec'])

# Plot historical prices
data = ct.download_data(tickers)
sg.plot_prices(data, tickers)

# Plot spread and trading signals
sg.plot_spread_and_signals(results['spread_vec'], signals_df, spread_mean, spread_std)
