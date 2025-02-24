import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

tickers = ['NVDA', 'TSM']

def download_data(tickers, start = '2015-01-01', end = '2025-01-01'):
    data = yf.download(tickers=tickers, start= start, end= end)['Close']
    data = data.dropna()
    return data

data = download_data(tickers)
print(data.columns)

plt.figure(figsize=(10,6))
plt.plot(data.index, data['NVDA'], label='NVDA')
plt.plot(data.index, data['TSM'], label='TSM')
plt.legend()
plt.title('Historical Prices of NVDA and TSM')
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

adf_test(data['NVDA'], 'NVDA')
adf_test(data['TSM'], 'TSM')

#Engle-Granger Two-Step method
score, pvalue, _ = coint(data['NVDA'], data['TSM'])
print("Engle-Granger Cointegration Test:")
print(f"  Test Statistic: {score:.4f}")
print(f"  p-value       : {pvalue:.4f}\n")

#Johanssen cointegration test (VECM)
johansen_result = coint_johansen(data[['NVDA', 'TSM']], det_order=0, k_ar_diff=1)
print("Johansen Cointegration Test Results:")
print("Eigenvalues:")
print(johansen_result.eig)
print("\nTrace Statistics:")
print(johansen_result.lr1)
print("\nCritical Values (90%, 95%, 99%):")
print(johansen_result.cvt)
