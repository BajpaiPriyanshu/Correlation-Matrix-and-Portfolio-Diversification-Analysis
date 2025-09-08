import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

print("=== CORRELATION MATRIX AND PORTFOLIO DIVERSIFICATION ANALYSIS ===\n")

print("STEP 1: Creating Simulated Stock Data")
print("-" * 40)

dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM']
n_stocks = len(stocks)

start_prices = np.array([150, 2800, 300, 200, 140, 170, 160, 90])

volatilities = np.array([0.02, 0.025, 0.018, 0.035, 0.02, 0.015, 0.012, 0.025])

correlation_base = np.random.randn(n_days, n_stocks)
market_factor = np.random.randn(n_days, 1)

returns = 0.7 * correlation_base + 0.3 * market_factor
returns = returns * volatilities

price_data = np.zeros((n_days, n_stocks))
price_data[0] = start_prices

for i in range(1, n_days):
    price_data[i] = price_data[i-1] * (1 + returns[i])

df_prices = pd.DataFrame(price_data, index=dates, columns=stocks)
print(f"Created price data for {n_stocks} stocks over {n_days} days")
print(f"Date range: {dates[0].date()} to {dates[-1].date()}\n")

print("STEP 2: Calculating Daily Returns")
print("-" * 40)

df_returns = df_prices.pct_change().dropna()
print("Daily returns calculated (percentage change from previous day)")
print(f"Returns data shape: {df_returns.shape}")
print(f"Sample returns for first 5 days:\n{df_returns.head()}\n")

print("STEP 3: Building Correlation Matrix")
print("-" * 40)

correlation_matrix = df_returns.corr()
print("Correlation Matrix:")
print(correlation_matrix.round(3))
print("\nInterpretation:")
print("1.0 = Perfect positive correlation (stocks move exactly together)")
print("0.0 = No correlation (stocks move independently)")
print("-1.0 = Perfect negative correlation (stocks move in opposite directions)\n")

print("STEP 4: Creating Correlation Heatmap")
print("-" * 40)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.8})

plt.title('Stock Correlation Matrix Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.show()
print("Heatmap created showing correlation relationships\n")

print("STEP 5: Finding Highly Correlated Stock Pairs")
print("-" * 40)

high_corr_threshold = 0.7
high_corr_pairs = []

for i in range(len(stocks)):
    for j in range(i+1, len(stocks)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > high_corr_threshold:
            high_corr_pairs.append((stocks[i], stocks[j], corr_value))

print(f"Highly correlated pairs (|correlation| > {high_corr_threshold}):")
if high_corr_pairs:
    for stock1, stock2, corr in high_corr_pairs:
        print(f"{stock1} - {stock2}: {corr:.3f}")
else:
    print("No highly correlated pairs found (good for diversification!)")
print()

print("STEP 6: Portfolio Risk and Diversification Analysis")
print("-" * 40)

equal_weights = np.array([1/n_stocks] * n_stocks)
print(f"Equal-weight portfolio: {dict(zip(stocks, equal_weights))}")

portfolio_returns = df_returns.dot(equal_weights)
print(f"Portfolio created with equal {1/n_stocks:.1%} allocation to each stock")

portfolio_mean_return = portfolio_returns.mean() * 252
portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = portfolio_mean_return / portfolio_volatility

print(f"\nPortfolio Performance Metrics:")
print(f"Annual Return: {portfolio_mean_return:.2%}")
print(f"Annual Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print(f"\nSTEP 7: Portfolio Variance Calculation")
print("-" * 40)

stock_volatilities = df_returns.std() * np.sqrt(252)
print("Individual Stock Volatilities (Annual):")
for stock, vol in zip(stocks, stock_volatilities):
    print(f"{stock}: {vol:.2%}")

covariance_matrix = df_returns.cov() * 252
portfolio_variance = np.dot(equal_weights.T, np.dot(covariance_matrix, equal_weights))
portfolio_std_dev = np.sqrt(portfolio_variance)

print(f"\nPortfolio Variance Calculation:")
print(f"Portfolio Variance: {portfolio_variance:.6f}")
print(f"Portfolio Standard Deviation: {portfolio_std_dev:.2%}")
print(f"Matches our earlier calculation: {abs(portfolio_std_dev - portfolio_volatility) < 0.001}")

print(f"\nSTEP 8: Diversification Benefit Analysis")
print("-" * 40)

avg_individual_volatility = stock_volatilities.mean()
print(f"Average Individual Stock Volatility: {avg_individual_volatility:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")

diversification_benefit = avg_individual_volatility - portfolio_volatility
diversification_ratio = portfolio_volatility / avg_individual_volatility

print(f"Diversification Benefit: {diversification_benefit:.2%}")
print(f"Risk Reduction Ratio: {diversification_ratio:.2f}")
print(f"Risk Reduced by: {(1 - diversification_ratio):.1%}")

print(f"\nSTEP 9: Creating Optimized Diversified Portfolio")
print("-" * 40)

avg_correlations = correlation_matrix.mean()
print("Average correlation of each stock with others:")
for stock, avg_corr in zip(stocks, avg_correlations):
    print(f"{stock}: {avg_corr:.3f}")

inverse_corr_weights = 1 / (avg_correlations + 0.1)
optimized_weights = inverse_corr_weights / inverse_corr_weights.sum()

print(f"\nOptimized Portfolio Weights:")
for stock, weight in zip(stocks, optimized_weights):
    print(f"{stock}: {weight:.1%}")

opt_portfolio_returns = df_returns.dot(optimized_weights)
opt_portfolio_volatility = opt_portfolio_returns.std() * np.sqrt(252)
opt_sharpe_ratio = (opt_portfolio_returns.mean() * 252) / opt_portfolio_volatility

print(f"\nOptimized Portfolio Performance:")
print(f"Annual Volatility: {opt_portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {opt_sharpe_ratio:.2f}")
print(f"Risk Reduction vs Equal Weight: {portfolio_volatility - opt_portfolio_volatility:.2%}")

print(f"\nSTEP 10: Final Analysis Summary")
print("=" * 50)

print("PORTFOLIO COMPARISON:")
print(f"Equal-Weight Portfolio:")
print(f"  - Volatility: {portfolio_volatility:.2%}")
print(f"  - Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Optimized Portfolio:")
print(f"  - Volatility: {opt_portfolio_volatility:.2%}")
print(f"  - Sharpe Ratio: {opt_sharpe_ratio:.2f}")
print(f"Improvement: {((opt_sharpe_ratio/sharpe_ratio - 1) * 100):+.1f}% better Sharpe ratio")

print(f"\nKEY INSIGHTS:")
print(f"1. Portfolio diversification reduced risk by {(1 - diversification_ratio):.1%}")
print(f"2. Correlation-based optimization further reduced risk by {((portfolio_volatility - opt_portfolio_volatility)/portfolio_volatility*100):.1f}%")
print(f"3. Average stock correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
print(f"4. Most correlated pair has correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max():.3f}")

print(f"\nCONCLUSION:")
print("This analysis demonstrates how correlation matrices help build diversified portfolios")
print("by identifying relationships between assets and optimizing risk-return profiles.")
print("\n" + "="*70)
print("ANALYSIS COMPLETE - Great job exploring quantitative finance!")
print("="*70)
