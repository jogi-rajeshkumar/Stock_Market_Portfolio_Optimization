<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Market Portfolio Optimization</title>
<style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        background-color: #f9f9f9;
        color: #333;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1, h2 {
        color: #007acc;
    }
    pre {
        background-color: #f4f4f4;
        border: 1px solid #ddd;
        border-left: 3px solid #007acc;
        color: #666;
        page-break-inside: avoid;
        font-family: monospace;
        font-size: 0.9em;
        line-height: 1.6;
        margin-bottom: 1.6em;
        max-width: 100%;
        overflow: auto;
        padding: 1em 1.5em;
        display: block;
        word-wrap: break-word;
    }
    code {
        background-color: #f4f4f4;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
        margin: 0;
        padding: 0.2em 0.4em;
    }
    img {
        max-width: 100%;
        height: auto;
    }
</style>
</head>
<body>
<div class="container">
    <h1>Stock Market Portfolio Optimization</h1>

    <h2>Installation</h2>
    <pre><code>pip install pandas yfinance matplotlib seaborn numpy</code></pre>

    <h2>Usage</h2>
    <p>Run the <code>main.py</code> script:</p>
    <pre><code>python main.py</code></pre>

    <h2>Code Overview</h2>

    <h2>Import Libraries</h2>
    <pre><code>import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np</code></pre>

    <h2>Define the Time Period and Stock Tickers</h2>
    <pre><code>end_date = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']</code></pre>

    <h2>Download Stock Data</h2>
    <pre><code>data = yf.download(tickers, start=start_date, end=end_date, progress=False)
data = data.reset_index() 
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]</code></pre>

    <h2>Melt and Pivot Data</h2>
    <pre><code>data_melted = data.melt(id_vars=['Date_'], var_name='Variable', value_name='Value')
data_melted[['Attribute', 'Ticker']] = data_melted['Variable'].str.rsplit('_', n=1, expand=True)
data_melted = data_melted.drop(columns=['Variable'])
data_pivoted = data_melted.pivot_table(index=['Date_', 'Ticker'], columns='Attribute', values='Value', aggfunc='first')
stock_data = data_pivoted.reset_index()
stock_data = stock_data.rename(columns={'Date_': 'Date'})
stock_data['Date'] = pd.to_datetime(stock_data['Date'])</code></pre>

    <h2>Visualize Adjusted Close Price Over Time</h2>
    <img src="https://user-images.githubusercontent.com/61099/242266547-63d98bd9-35f3-4dfe-92f4-a4a8dd75aa5c.png" alt="Stock Price Data">
    <pre><code>sns.set(style='whitegrid')
plt.figure(figsize=(14, 7))
sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o')
plt.title('Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend(title='Ticker')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()</code></pre>

    <h2>Calculate and Plot Moving Averages</h2>
    <pre><code>short_window = 50
long_window = 200
stock_data.set_index('Date', inplace=True)
unique_tickers = stock_data['Ticker'].unique()

for ticker in unique_tickers:
    ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
    ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
    ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_data.index, ticker_data['Adj Close'], label='Adj Close')
    plt.plot(ticker_data.index, ticker_data['50_MA'], label='50-Day MA')
    plt.plot(ticker_data.index, ticker_data['200_MA'], label='200-Day MA')
    plt.title(f'{ticker} - Adjusted Close and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.bar(ticker_data.index, ticker_data['Volume'], label='Volume', color='orange')
    plt.title(f'{ticker} - Volume Traded')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()</code></pre>

    <h2>Calculate Daily Returns and Plot Distribution</h2>
    <pre><code>stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')

for ticker in unique_tickers:
    ticker_data = stock_data[stock_data['Ticker'] == ticker]
    sns.histplot(ticker_data['Daily Return'].dropna(), bins=50, kde=True, label=ticker, alpha=0.5)

plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend(title='Ticker')
plt.grid(True)
plt.tight_layout()
plt.show()</code></pre>

    <h2>Correlation Matrix of Daily Returns</h2>
    <pre><code>daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
correlation_matrix = daily_returns.corr()

plt.figure(figsize=(12, 10))
sns.set(style='whitegrid')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f', annot_kws={"size": 10})
plt.title('Correlation Matrix of Daily Returns')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()</code></pre>

    <h2>Calculate Expected Returns and Volatility</h2>
    <pre><code>expected_returns = daily_returns.mean() * 252  ## annualize the returns
volatility = daily_returns.std() * np.sqrt(252)  ## annualize the volatility

stock_stats = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})</code></pre>

    <h2>Portfolio Optimization</h2>
    <pre><code>def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

num_portfolios = 10000
results = np.zeros((3, num_portfolios))
cov_matrix = daily_returns.cov() * 252
np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(unique_tickers))
    weights /= np.sum(weights)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = portfolio_return / portfolio_volatility  ## Sharpe Ratio

plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.show()</code></pre>

    <h2>Maximum Sharpe Ratio Portfolio</h2>
    <pre><code>max_sharpe_idx = np.argmax(results[2])
max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_volatility = results[1, max_sharpe_idx]
max_sharpe_ratio = results[2, max_sharpe_idx]

max_sharpe_weights = np.zeros(len(unique_tickers))

for i in range(num_portfolios):
    weights = np.random.random(len(unique_tickers))
    weights /= np.sum(weights)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

    if results[2, i] == max_sharpe_ratio:
        max_sharpe_weights = weights
        break

portfolio_weights_df = pd.DataFrame({
    'Ticker': unique_tickers,
    'Weight': max_sharpe_weights
})
portfolio_weights_df</code></pre>
</div>
</body>
</html>
