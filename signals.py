import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_prices(data, tickers):
    """Plots historical prices of the given tickers."""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[tickers[0]], label=tickers[0])
    plt.plot(data.index, data[tickers[1]], label=tickers[1])
    plt.legend()
    plt.title(f'Historical Prices of {tickers[0]} and {tickers[1]}')
    plt.show()

def generate_signals(spread_vec):
    """Generates trading signals based on the spread."""
    spread_mean = spread_vec.mean()
    spread_std = spread_vec.std()
    
    signal = np.where(spread_vec > spread_mean + 1.5 * spread_std, -1,
                      np.where(spread_vec < spread_mean - 1.5 * spread_std, 1, 0))
    
    signals_df = pd.DataFrame({'Spread': spread_vec, 'Signal': signal}, index=spread_vec.index)
    return signals_df, spread_mean, spread_std

def plot_spread_and_signals(spread_vec, signals_df, spread_mean, spread_std):
    """Plots the spread with threshold levels and signals."""
    plt.figure(figsize=(10, 6))
    plt.plot(spread_vec.index, spread_vec, label='Spread')
    plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
    plt.axhline(spread_mean + 1.5 * spread_std, color='red', linestyle='--', label='+1.5 STD')
    plt.axhline(spread_mean - 1.5 * spread_std, color='green', linestyle='--', label='-1.5 STD')
    plt.legend()
    plt.title('Cointegration Spread and Threshold Levels')
    plt.show()

    # Plot trading signals
    plt.figure(figsize=(10, 6))
    plt.plot(spread_vec.index, spread_vec, label='Spread')
    plt.scatter(spread_vec.index[signals_df['Signal'] == 1], spread_vec[signals_df['Signal'] == 1], marker='^', color='green', label='Long Signal', s=100)
    plt.scatter(spread_vec.index[signals_df['Signal'] == -1], spread_vec[signals_df['Signal'] == -1], marker='v', color='red', label='Short Signal', s=100)
    plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
    plt.axhline(spread_mean + 1.5 * spread_std, color='red', linestyle='--', label='+1.5 STD')
    plt.axhline(spread_mean - 1.5 * spread_std, color='green', linestyle='--', label='-1.5 STD')
    plt.legend()
    plt.title('Trading Signals Based on Spread')
    plt.show()
