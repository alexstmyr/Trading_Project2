import cointegration_test as ct
import signals as sg
import kalman_filter as kf
import backtesting as bt
import pandas as pd


# Define parameters.
# We model: MSFT = const + β * AMD.
tickers = ['AMD', 'QCOM']
initial_capital = 1_000_000   # Initial capital.
n_shares = 200                # Number of shares per trade.
commission = 0.125 / 100      # Commission per trade.

# --- COINTEGRATION TEST ---
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

# --- Johansen Test Results ---
beta_x, beta_y = results["johansen_beta"]
print("\n=== Johansen Test Results ===")
print(f"Estimated Johansen beta vector: (beta_x: {beta_x:.4f}, beta_y: {beta_y:.4f})")

# --- DYNAMIC HEDGE RATIO VIA KALMAN FILTER ---
# Our OLS regression is: MSFT = const + β * AMD, so for dynamic estimation:
#   independent = data["AMD"], dependent = data["MSFT"]
kf_model = kf.KalmanFilterReg()
dynamic_hedge_ratios = kf_model.run_kalman_filter(data[tickers[1]].values, data[tickers[0]].values)

# Compute dynamic spread using the dynamic hedge ratio:
# dynamic_spread = MSFT - (dynamic β) * AMD.
dynamic_spread = data[tickers[0]] - dynamic_hedge_ratios * data[tickers[1]]
dyn_mean = dynamic_spread.mean()
dyn_std = dynamic_spread.std()
normalized_dynamic_spread = (dynamic_spread - dyn_mean) / dyn_std

hedge_ratios_df = pd.DataFrame({
    "Date": data.index,
    "Dynamic Hedge Ratio": dynamic_hedge_ratios
}).set_index("Date")
print("\n=== Dynamic Hedge Ratios (First 10 Values) ===")
print(hedge_ratios_df)

# --- SIGNAL GENERATION ---
# Generate signals using the dynamic spread.
# (generate_signals_with_indices returns a DataFrame with columns:
# 'Normalized Spread', 'Signal_Asset1', 'Signal_Asset2')
signals_df, long_signals, short_signals, close_signals, spread_mean, spread_std = sg.generate_signals_with_indices(dynamic_spread)

# --- BACKTESTING ---
portfolio_df, init_val, final_val, trades, win_rate = bt.run_backtest(data, signals_df, initial_capital, n_shares, commission)

# --- PLOTTING ---
sg.plot_combined_chart(data, tickers, dynamic_spread, signals_df)

# --- PRINT BACKTEST RESULTS ---
print("\n=== Backtesting Results ===")
print(f"Initial Portfolio Value: ${init_val:,.2f}")
print(f"Final Portfolio Value:   ${final_val:,.2f}")
print(f"Number of Trades:        {trades}")
print(f"Win Rate:                {win_rate:.2f}%")
