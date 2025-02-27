import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import backtesting as bt
import pandas as pd

# Define parameters.
tickers = ['AMD', 'QCOM']
initial_capital = 1_000_000   # Initial capital.
n_shares = 200                # Number of shares per trade.
commission = 0.125 / 100      # Commission per trade.

# Run cointegration test and get results.
results = ct.coint_test(tickers)
data = results["data"]

# Print ADF Test Results.
print("\n=== ADF Test Results ===")
for ticker, result in results["adf_results"].items():
    print(f"\nADF Test for {ticker}:")
    print(f"  Test Statistic : {result['Test Statistic']:.4f}")
    print(f"  p-value        : {result['p-value']:.4f}")
    print(f"  # Lags Used    : {result['# Lags Used']}")
    print("  Critical Values:")
    for key, value in result["Critical Values"].items():
        print(f"    {key}: {value:.4f}")

print("\n=== OLS Regression Summary ===")
print(results["ols_summary"])

# Print Engle-Granger Cointegration Test Results.
print("\n=== Engle-Granger Cointegration Test ===")
print(f"  Test Statistic: {results['coint_test']['Test Statistic']:.4f}")
print(f"  p-value       : {results['coint_test']['p-value']:.4f}")

# Apply Kalman Filter for Dynamic Hedge Ratio Estimation.
kf_model = kf.KalmanFilterReg()
hedge_ratios = kf_model.run_kalman_filter(data[tickers[1]], data[tickers[0]])

# Create a DataFrame for the dynamic hedge ratios.
hedge_ratios_df = pd.DataFrame({
    "Date": data.index,
    "Hedge Ratio": hedge_ratios
}).set_index("Date")

print("\n=== Dynamic Hedge Ratios (First 10 Values) ===")
print(hedge_ratios_df.head(10))

# Compute spread using Johansen Cointegration.
beta_x, beta_y = results["johansen_beta"]
spread_vec = beta_x * data[tickers[1]] + beta_y * data[tickers[0]]

# Generate Trading Signals based on the Johansen spread.
signals_df, spread_mean, spread_std = sg.generate_signals(spread_vec)

# Run Backtesting.
portfolio_df, init_val, final_val, trades, win_rate = bt.run_backtest(data, signals_df, initial_capital, n_shares, commission)

# Plot the combined chart with asset prices and signals.
sg.plot_combined_chart(data, tickers, spread_vec, signals_df)

# Print backtesting results.
print("\n=== Backtesting Results ===")
print(f"Initial Portfolio Value: ${init_val:,.2f}")
print(f"Final Portfolio Value:   ${final_val:,.2f}")
print(f"Number of Trades:        {trades}")
print(f"Win Rate:                {win_rate:.2f}%")
