import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

df = pd.read_excel('updated-annual-vol-returns.xlsx', sheet_name = 'data')
print("Column Names:", df.columns)
returns_columns = df.columns[1:]
results = {}

for col in returns_columns:
    data = df[col].dropna()  # Drop missing values
    shapiro_p = stats.shapiro(data)[1]  # Shapiro-Wilk test
    jarque_bera_p = stats.jarque_bera(data)[1]  # Jarque-Bera test

    results[col] = {
        "Shapiro-Wilk p-value": shapiro_p,
        "Jarque-Bera p-value": jarque_bera_p
    }

    # Q-Q Plot
    plt.figure(figsize=(6,4))
    sm.qqplot(data, line='s')
    plt.title(f"{col} \n Q-Q Plot vs Normal")
    plt.savefig(f"{col}_qqplot.png", dpi=300, bbox_inches='tight')

    # ACF Plot
    plt.figure(figsize=(6,4))
    plot_acf(data, fft=False)
    plt.title(f"{col} \n ACF for Original Values")
    plt.savefig(f"{col}_acf.png", dpi=300, bbox_inches='tight')

    # ACF Plot for absolute values
    plt.figure(figsize=(6,4))
    plot_acf(np.abs(data), fft=False)
    plt.title(f"{col} \n ACF for Absolute Values")
    plt.savefig(f"{col}_absacf.png", dpi=300, bbox_inches='tight')

# Print normality test results
for col, test_results in results.items():
    print(f"\nResults for {col}:")
    for test, p_value in test_results.items():
        print(f"{test}: {p_value:.5f}")

