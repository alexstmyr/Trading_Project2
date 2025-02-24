import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

tickers = ['NVDA', 'TSM']

def coint_test(tickers):

    def download_data(tickers, start = '2015-01-01', end = '2025-01-01'):
        data = yf.download(tickers=tickers, start= start, end= end)['Close']
        data = data.dropna()
        return data

    data = download_data(tickers)

    plt.figure(figsize=(10,6))
    plt.plot(data.index, data[tickers[0]], label=tickers[0])
    plt.plot(data.index, data[tickers[1]], label=tickers[1])
    plt.legend()
    plt.title(f'Historical Prices of {tickers[0]} and {tickers[1]}')
    plt.show()

    #ADF test
    def adf_test(series, series_name):
        result = adfuller(series)
        print(f"ADF Test for {series_name}:")
        print(f"  Test Statistic : {result[0]:.4f}")
        print(f"  p-value        : {result[1]:.4f}")
        print(f"  # Lags Used   : {result[2]}")
        print("  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.4f}")
        print("------\n")

    adf_test(data[tickers[0]], tickers[0])
    adf_test(data[tickers[1]], tickers[1])

    #OLS Regression
    X = sm.add_constant(data[tickers[1]])
    y = data[tickers[0]]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Compute the residuals (spread) from the regression.
    spread = model.resid

    # Plot the residuals to visualize the spread.
    plt.figure(figsize=(10,6))
    plt.plot(spread.index, spread, label='Spread (Residuals)')
    plt.legend()
    plt.title(f'Cointegration Spread from OLS Regression ({tickers[0]} ~ {tickers[1]})')
    plt.show()

    # Test if the spread is stationary using the ADF test.
    adf_test(spread, 'Spread')

    #Engle-Granger Two-Step method
    score, pvalue, _ = coint(data[tickers[0]], data[tickers[1]])
    print("Engle-Granger Cointegration Test:")
    print(f"  Test Statistic: {score:.4f}")
    print(f"  p-value       : {pvalue:.4f}\n")

    #Johanssen cointegration test (VECM)
    johansen_result = coint_johansen(data[[tickers[0], tickers[1]]], det_order=0, k_ar_diff=1)
    print("Johansen Cointegration Test Results:")
    print("Eigenvalues:")
    print(johansen_result.eig)
    print("\nTrace Statistics:")
    print(johansen_result.lr1)
    print("\nCritical Values (90%, 95%, 99%):")
    print(johansen_result.cvt)
