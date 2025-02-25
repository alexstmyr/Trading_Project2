import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

def download_data(tickers, start='2015-01-01', end='2025-01-01'):
    """Downloads historical price data."""
    data = yf.download(tickers=tickers, start=start, end=end)['Close']
    return data.dropna()

def adf_test(series, series_name):
    """Performs the Augmented Dickey-Fuller test."""
    result = adfuller(series)
    return {
        "Test Statistic": result[0],
        "p-value": result[1],
        "# Lags Used": result[2],
        "Critical Values": result[4]
    }

def coint_test(tickers):
    """Performs cointegration tests on a pair of assets."""
    data = download_data(tickers)
    
    # Perform ADF test on individual assets
    adf_results = {ticker: adf_test(data[ticker], ticker) for ticker in tickers}
    
    # OLS Regression
    X = sm.add_constant(data[tickers[1]])
    y = data[tickers[0]]
    model = sm.OLS(y, X).fit()
    spread = model.resid  # Residuals
    
    # ADF Test on Residuals (Engle-Granger)
    spread_adf_result = adf_test(spread, 'Spread')
    
    # Engle-Granger Cointegration Test
    score, pvalue, _ = coint(data[tickers[0]], data[tickers[1]])
    
    # Johansen Cointegration Test
    johansen_result = coint_johansen(data[[tickers[0], tickers[1]]], det_order=1, k_ar_diff=1)
    beta = johansen_result.evec[:, 0]
    beta = beta / beta[0]
    
    spread_vec = data[tickers[0]] + beta[1] * data[tickers[1]]
    
    return {
        "adf_results": adf_results,
        "spread_adf": spread_adf_result,
        "coint_test": {"Test Statistic": score, "p-value": pvalue},
        "johansen_beta": beta,
        "spread_vec": spread_vec
    }
