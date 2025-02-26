import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def download_data(tickers, start='2015-01-01', end='2025-01-01'):
    """Downloads historical price data for the given tickers."""
    data = yf.download(tickers=tickers, start=start, end=end)['Close']
    return data.dropna()

def adf_test(series, series_name):
    """Performs the Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(series)
    return {
        "Test Statistic": result[0],
        "p-value": result[1],
        "# Lags Used": result[2],
        "Critical Values": result[4]
    }

def coint_test(tickers):
    """Performs cointegration tests on a pair of assets and calculates the spread."""
    data = download_data(tickers)
    
    # ADF Test on individual assets
    adf_results = {ticker: adf_test(data[ticker], ticker) for ticker in tickers}
    
    # OLS Regression to get hedge ratio:
    X = sm.add_constant(data[tickers[1]])  # Independent variable
    y = data[tickers[0]]  # Dependent variable
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params[tickers[1]]  # Hedge ratio from regression

    ols_summary = model.summary().as_text()

    # Compute residual spread (Engle-Granger)
    spread = model.resid
    spread_adf_result = adf_test(spread, 'Spread')
    
    # Engle-Granger Cointegration Test (using the same ordering)
    score, pvalue, _ = coint(data[tickers[1]], data[tickers[0]])
    
    # Johansen Cointegration Test
    johansen_result = coint_johansen(data[[tickers[1], tickers[0]]], det_order=0, k_ar_diff=1)
    beta_x = johansen_result.evec[0, 0]
    beta_y = johansen_result.evec[1, 0]

    return {
        "adf_results": adf_results,
        "spread_adf": spread_adf_result,
        "ols_summary": ols_summary,
        "coint_test": {"Test Statistic": score, "p-value": pvalue},
        "johansen_beta": (beta_x, beta_y),
        "hedge_ratio": hedge_ratio,
        "data": data
    }
