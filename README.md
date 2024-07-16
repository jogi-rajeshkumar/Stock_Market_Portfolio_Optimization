# Stock_Market_Portfolio_Optimization

Getting Started with Stock Market Portfolio Optimization
To achieve diversification and optimal risk-return trade-offs, stock market portfolio optimization involves analyzing price trends, calculating expected returns and volatilities, and assessing correlations between stocks. Modern Portfolio Theory (MPT) is instrumental in constructing efficient portfolios that balance risk and return along the efficient frontier.

The goal of stock market portfolio optimization is to identify portfolios with the highest Sharpe ratio, indicating superior risk-adjusted returns and offering clear allocation strategies to meet long-term investment objectives.

To begin, we gather real-time stock market data using the yfinance API.


Stock Market Portfolio Optimization with Python
Let's initiate stock market portfolio optimization by importing necessary Python libraries and fetching data using the yfinance API. If you're new to using this API, install it in your Python environment with the following command:

## Project Structure

- `main.py`: Main script that downloads stock data, performs analysis, and visualizes the results.
- `README.md`: This file.
- Output files 

## Dependencies

- pandas
- yfinance
- matplotlib
- seaborn
- numpy

You can install the required libraries using pip:

```sh
pip install pandas yfinance matplotlib seaborn numpy


## Usage
Run the main.py script: python main.py


## Code Overview
## Import Libraries
```
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Next, let's collect data for some popular Indian companies:
```
## Define the Time Period and Stock Tickers
end_date = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

## Download Stock Data
data = yf.download(tickers, start=start_date, end=end_date, progress=False)
data = data.reset_index() 
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]


```
![Stock Market Portfolio Optimization](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Capture.PNG)


## Melt and Pivot Data
```

data_melted = data.melt(id_vars=['Date_'], var_name='Variable', value_name='Value')
data_melted[['Attribute', 'Ticker']] = data_melted['Variable'].str.rsplit('_', n=1, expand=True)
data_melted = data_melted.drop(columns=['Variable'])
data_pivoted = data_melted.pivot_table(index=['Date_', 'Ticker'], columns='Attribute', values='Value', aggfunc='first')
stock_data = data_pivoted.reset_index()
stock_data = stock_data.rename(columns={'Date_': 'Date'})
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

## Visualize Adjusted Close Price![Uploading Capture.PNG…]()
 Over Time
sns.set(style='whitegrid')
plt.figure(figsize=(14, 7))
sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o')
plt.title('Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend(title='Ticker')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

```
![Figure 1](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_1.png)

We'll examine the stock market performance of these companies from July 2023 to July 2024:

Visualizing the adjusted close prices of HDFCBANK.NS, INFY.NS, RELIANCE.NS, and TCS.NS reveals that TCS exhibits the highest adjusted close prices, followed by RELIANCE, INFY, and HDFCBANK. RELIANCE and TCS show strong upward trends, while HDFCBANK and INFY demonstrate more stability with fewer price fluctuations.

Let's compute the 50-day and 200-day moving averages and plot them alongside the Adjusted Close price for each stock:
```

## Calculate and Plot Moving Averages
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

```
![Figure 2](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_2.png)

![Figure 3](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_3.png)

![Figure 4](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_4.png)

![Figure 5](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_5.png)

![Figure 6](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_6.png)

![Figure 7](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_7.png)

![Figure 8](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_8.png)

![Figure 9](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_9.png)

While HDFCBANK and INFY initially declined, they later showed signs of recovery. In contrast, RELIANCE and TCS maintained consistent upward trends in adjusted close prices. Volume traded graphs indicate significant trading activity, particularly notable in HDFCBANK and RELIANCE around early 2024. These insights aid in informed investment decision-making by understanding price movements and trading behaviors.

Next, we examine the distribution of daily returns for these stocks:

```

## Calculate Daily Returns and Plot Distribution
stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

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
plt.show()

```
![Figure 10](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_10.png)

The distributions approximate normality, centered around zero, suggesting that most daily returns cluster near the average. However, there are tails on both ends, reflecting occasional significant gains or losses. INFY and RELIANCE exhibit slightly wider distributions, indicating higher volatility compared to HDFCBANK and TCS.

Let's explore the correlation between these stocks:
```

## Correlation Matrix of Daily Returns
daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
correlation_matrix = daily_returns.corr()

plt.figure(figsize=(12, 10))
sns.set(style='whitegrid')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f', annot_kws={"size": 10})
plt.title('Correlation Matrix of Daily Returns')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

```
![Figure 11](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_11.png)

INFY and TCS exhibit a high positive correlation (0.71), indicating they tend to move in tandem. HDFCBANK shows moderate positive correlations with RELIANCE (0.37) and lower correlations with INFY (0.17) and TCS (0.10). RELIANCE displays low correlations with INFY (0.19) and TCS (0.13). These varying correlations suggest potential diversification benefits, as combining stocks with lower correlations can reduce overall portfolio risk.

Portfolio Optimization
Using Modern Portfolio Theory, we construct efficient portfolios by balancing risk and return. Our approach includes:

 -Calculating expected returns and volatility for each stock.
 
 -Generating random portfolios to identify the efficient frontier.
 
 -Optimizing the portfolio to maximize the Sharpe ratio.

First, let's compute expected returns and volatility for each stock:
```

## Calculate Expected Returns and Volatility
expected_returns = daily_returns.mean() * 252  ## annualize the returns
volatility = daily_returns.std() * np.sqrt(252)  ## annualize the volatilit

stock_stats = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})

```
![Capture1](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Capture1.PNG)

RELIANCE shows the highest expected return (29.73%) with moderate volatility (21.47%), indicating potential high-reward, higher-risk investment. INFY and TCS also offer high expected returns (21.38% and 22.09%, respectively) with moderate volatility (23.23% and 19.69%). HDFCBANK presents the lowest expected return (1.37%) with moderate volatility (20.69%), making it less attractive in terms of risk-adjusted returns.

Next, we generate multiple random portfolios to plot the efficient frontier:

```

## Portfolio Optimization
def portfolio_performance(weights, returns, cov_matrix):
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
plt.show()

```
![Figure 12](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Figure_12.png)

Each dot on the plot represents a portfolio, with color denoting the Sharpe ratio—a measure of risk-adjusted return. Portfolios on the leftmost edge offer the highest expected returns for a given level of volatility, representing optimal portfolios. Darker blue indicates portfolios with higher Sharpe ratios, signifying better risk-adjusted returns.

Identifying the portfolio with the maximum Sharpe ratio reveals:
```

## Maximum Sharpe Ratio Portfolio
max_sharpe_idx = np.argmax(results[2])
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
portfolio_weights_df

```
![Capture2](https://github.com/jogi-rajeshkumar/Stock_Market_Portfolio_Optimization/blob/main/output/Capture2.PNG)

Expected Return:

HDFCBANK.NS: 0.0237
INFY.NS: 0.2077
RELIANCE.NS: 0.2242
TCS.NS: 0.2224
Volatility:

HDFCBANK.NS: 0.2113
INFY.NS: 0.2271
RELIANCE.NS: 0.2103
TCS.NS: 0.2012
Sharpe Ratio: (To calculate Sharpe Ratio, we need the risk-free rate, assuming it's zero for this example.)

Let's detail the allocation of stocks in this portfolio:

The portfolio is diversified with allocations as follows:

HDFCBANK.NS: 21.63%
INFY.NS: 31.96%
RELIANCE.NS: 11.96%
TCS.NS: 34.45%
This allocation indicates that TCS.NS has the largest allocation, contributing significantly to portfolio performance, while INFY.NS has the smallest allocation. This balanced distribution aims to maximize returns while managing risk through diversified exposure to different stocks.

Summary
Stock market portfolio optimization involves analyzing price trends, expected returns, volatilities, and correlations to achieve diversification and optimize risk-adjusted returns.

I hope you find this article on stock market portfolio optimization with Python informative. Feel free to ask questions in the comments section below.
```
