import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Define the time period for the data
end_date = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# List of stock tickers to download
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

# Download the stock data
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Reset index to bring Date into the columns
data = data.reset_index()

# Flatten the MultiIndex columns
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

# Print the columns and the first few rows to check if 'Date_' is present
print("Columns after flattening index:", data.columns)
print("First few rows of data:\n", data.head())

# Melt the DataFrame to make it long format where each row is a unique combination of Date_, Ticker, and attributes
data_melted = data.melt(id_vars=['Date_'], var_name='Variable', value_name='Value')

# Split the 'Variable' column into 'Attribute' and 'Ticker'
data_melted[['Attribute', 'Ticker']] = data_melted['Variable'].str.rsplit('_', n=1, expand=True)

# Drop the 'Variable' column
data_melted = data_melted.drop(columns=['Variable'])

# Pivot the melted DataFrame to have the attributes (Open, High, Low, etc.) as columns
data_pivoted = data_melted.pivot_table(index=['Date_', 'Ticker'], columns='Attribute', values='Value', aggfunc='first')

# Reset index to turn multi-index into columns
stock_data = data_pivoted.reset_index()

# Rename 'Date_' to 'Date'
stock_data = stock_data.rename(columns={'Date_': 'Date'})

# Ensure 'Date' column is present
if 'Date' in stock_data.columns:
    # Convert 'Date' to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Plotting
    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 7))

    # Line plot for Adjusted Close Price over time for each ticker
    sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o')

    plt.title('Adjusted Close Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adjusted Close Price', fontsize=14)
    plt.legend(title='Ticker', title_fontsize='13', fontsize='11')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    # Calculate moving averages
    short_window = 50
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
        plt.show()
else:
    print("Error: 'Date' column is not present in the DataFrame.")

stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')

for ticker in unique_tickers:
    ticker_data = stock_data[stock_data['Ticker'] == ticker]
    sns.histplot(ticker_data['Daily Return'].dropna(), bins=50, kde=True, label=ticker, alpha=0.5)

plt.title('Distribution of Daily Returns', fontsize=16)
plt.xlabel('Daily Return', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Ticker', title_fontsize='13', fontsize='11')
plt.grid(True)
plt.tight_layout()
plt.show()

daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
correlation_matrix = daily_returns.corr()

plt.figure(figsize=(12, 10))
sns.set(style='whitegrid')

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f', annot_kws={"size": 10})
plt.title('Correlation Matrix of Daily Returns', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

import numpy as np

expected_returns = daily_returns.mean() * 252  # annualize the returns
volatility = daily_returns.std() * np.sqrt(252)  # annualize the volatility

stock_stats = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})

print (stock_stats)

# function to calculate portfolio performance
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# number of portfolios to simulate
num_portfolios = 10000

# arrays to store the results
results = np.zeros((3, num_portfolios))

# annualized covariance matrix
cov_matrix = daily_returns.cov() * 252

np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(unique_tickers))
    weights /= np.sum(weights)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.show()

max_sharpe_idx = np.argmax(results[2])
max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_volatility = results[1, max_sharpe_idx]
max_sharpe_ratio = results[2, max_sharpe_idx]

max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio


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

print (portfolio_weights_df)
