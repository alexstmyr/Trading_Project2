import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import backtesting as bt
import pandas as pd

# ----- PARAMETERS -----
tickers = ['AMD', 'QCOM']
initial_capital = 1_000_000   # Starting capital.
n_shares = 1000                # Number of shares per trade.
commission = 0.125 / 100      # Commission per trade (0.125%).

# ----- GROUP 1: COINTEGRATION & DYNAMIC HEDGE RATIO -----
# Run cointegration tests and retrieve price data.
results = ct.coint_test(tickers)
data = results["data"]

print("\n=== ADF Test Results ===")
for ticker, res in results["adf_results"].items():
    print(f"\nADF Test for {ticker}:")
    print(f"  Test Statistic : {res['Test Statistic']:.4f}")
    print(f"  p-value        : {res['p-value']:.4f}")
    print(f"  # Lags Used    : {res['# Lags Used']}")
    print("  Critical Values:")
    for key, value in res["Critical Values"].items():
        print(f"    {key}: {value:.4f}")

print("\n=== OLS Regression Summary ===")
print(results["ols_summary"])
print("\n=== Engle-Granger Cointegration Test ===")
print(f"  Test Statistic: {results['coint_test']['Test Statistic']:.4f}")
print(f"  p-value       : {results['coint_test']['p-value']:.4f}")

beta_ols = results["beta"]
print(f"\nStatic OLS Hedge Ratio: {beta_ols:.4f}")

# Print Johansen test results.
beta_x, beta_y = results["johansen_beta"]
print("\n=== Johansen Test Results ===")
print(f"Estimated Johansen beta vector: (beta_x: {beta_x:.4f}, beta_y: {beta_y:.4f})")

# Apply Kalman Filter for dynamic hedge ratio estimation.
kf_model = kf.KalmanFilterReg()
dynamic_hedge_ratios = kf_model.run_kalman_filter(data[tickers[0]].values, data[tickers[1]].values)

# Compute dynamic spread using the dynamic hedge ratio.
# dynamic_spread = MSFT - (dynamic β)*AMD.
dynamic_spread = beta_x * data[tickers[0]] + beta_y * data[tickers[1]]

# Save dynamic hedge ratios for reference.
hedge_ratios_df = pd.DataFrame({
    "Date": data.index,
    "Dynamic Hedge Ratio": dynamic_hedge_ratios
}).set_index("Date")
print("\n=== Dynamic Hedge Ratios (First 10 Values) ===")
print(hedge_ratios_df)

# ----- GROUP 2: SIGNALS & BACKTESTING -----
# Generate trading signals based on the dynamic spread.
signals_df, spread_mean, spread_std = sg.generate_signals(dynamic_spread)

#Add dynamic hedge-ratio to the signals data frame
signals_df["Dynamic Hedge Ratio"] = pd.Series(dynamic_hedge_ratios, index=data.index)

# Run backtesting.
portfolio_df, init_val, final_val, total_trades, win_rate, trades_df = bt.run_backtest(
    data, signals_df, initial_capital, n_shares, commission
)

# Plot the strategy chart:
# Upper panel: Price series for MSFT and AMD with trade markers on their respective axes.
# Lower panel: Normalized spread with ±1.5σ boundaries and zero line.
sg.plot_strategy(data, tickers, dynamic_spread, signals_df)

# Print backtesting results.
print("\n=== Backtesting Results ===")
print(f"Initial Portfolio Value: ${init_val:,.2f}")
print(f"Final Portfolio Value:   ${final_val:,.2f}")
print(f"Number of Trades:        {total_trades}")
print(f"Win Rate:                {win_rate:.2f}%")
print("\n=== Trade Log ===")
print(trades_df)
