import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import backtesting as bt
import pandas as pd

# ----- PARAMETERS -----
# We model: MSFT = const + β * AMD (i.e. dependent: MSFT, independent: AMD).
tickers = ['MSFT', 'AMD']
initial_capital = 1_000_000   # Starting capital.
n_shares = 200                # Number of shares per trade.
commission = 0.125 / 100      # Commission per trade (0.125%).

# ----- GROUP 1: COINTEGRATION & DYNAMIC HEDGE RATIO -----
# Run cointegration tests to retrieve the price data.
results = ct.coint_test(tickers)
data = results["data"]

# Print cointegration test results.
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
# (For our model: dependent = MSFT (tickers[0]), independent = AMD (tickers[1]).)
kf_model = kf.KalmanFilterReg()
dynamic_hedge_ratios = kf_model.run_kalman_filter(data[tickers[1]].values, data[tickers[0]].values)

# Compute dynamic spread using the dynamic hedge ratio.
# dynamic_spread = MSFT - (dynamic β)*AMD.
dynamic_spread = beta_x * data[tickers[0]] + beta_y * data[tickers[1]]

# Create a DataFrame for reference.
hedge_ratios_df = pd.DataFrame({
    "Date": data.index,
    "Dynamic Hedge Ratio": dynamic_hedge_ratios
}).set_index("Date")
print("\n=== Dynamic Hedge Ratios (First 10 Values) ===")
print(hedge_ratios_df.head(10))

# ----- GROUP 2: SIGNALS & BACKTESTING -----
# Generate trading signals based on the dynamic spread.
# (Signals are generated with a rule: long when normalized spread < -1.5,
#  short when normalized spread > 1.5, and exit when it returns to 0.)
signals_df, spread_mean, spread_std = sg.generate_signals(dynamic_spread)

# Run backtesting. The strategy opens trades when the signal goes to ±1 and closes all positions when the signal returns to 0.
portfolio_df, init_val, final_val, trades, win_rate = bt.run_backtest(data, signals_df, initial_capital, n_shares, commission)

# Plot the strategy chart:
#   Upper panel: price series for MSFT and AMD with trade markers.
#   Lower panel: normalized dynamic spread with boundaries.
sg.plot_strategy(data, tickers, dynamic_spread, signals_df)

# Print backtesting results.
print("\n=== Backtesting Results ===")
print(f"Initial Portfolio Value: ${init_val:,.2f}")
print(f"Final Portfolio Value:   ${final_val:,.2f}")
print(f"Number of Trades:        {trades}")
print(f"Win Rate:                {win_rate:.2f}%")
