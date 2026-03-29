# Portfolio Risk Analyser — NSE Stocks

An interactive portfolio risk analysis dashboard built with 
Python and Streamlit, applying Modern Portfolio Theory and 
Monte Carlo simulation to 5 NSE-listed Indian equities.

## Live Demo
[Open Live App](https://portfolio-risk-analyser-mfkc5smwc8w67uq3yomq6s.streamlit.app)

## What This Project Does

- Downloads 3 years of real NSE stock data (2022-2024)
- Analyses daily return distributions across 5 sectors
- Computes correlation matrix to measure diversification benefit
- Runs 10,000 portfolio simulations to plot Efficient Frontier
- Identifies Maximum Sharpe and Minimum Volatility portfolios
- Runs Monte Carlo simulation (1000 paths) on optimal portfolio
- Calculates VaR and CVaR at 95% and 99% confidence levels

## Stocks Analysed

| Stock | Sector |
|---|---|
| TCS.NS | Information Technology |
| HDFCBANK.NS | Banking |
| HINDUNILVR.NS | FMCG |
| MARUTI.NS | Automobile |
| RELIANCE.NS | Energy |

## Key Results

| Metric | Value |
|---|---|
| Median Portfolio Value (1 year) | ₹1,11,998 |
| VaR 95% | ₹85,153 |
| VaR 99% | ₹76,248 |
| CVaR 95% | ₹80,170 |
| Optimal Portfolio Volatility | 17% |
| Individual Stock Volatility | 20-22% |

## Business Insight

Combining 5 low-correlation NSE stocks reduces portfolio 
volatility from 20-22% to 17% — demonstrating the core 
benefit of diversification through Modern Portfolio Theory.

HINDUNILVR dominates the optimal portfolio at 63.65% 
allocation due to its combination of lowest volatility 
and lowest correlation with other holdings — consistent 
with FMCG's defensive characteristics.

All pairwise correlations between 0.20-0.35 confirm 
genuine diversification benefit across sectors.

The Maximum Sharpe portfolio delivers 12.5% annual return 
at 17% volatility — significantly better risk-adjusted 
performance than any individual stock in the portfolio.

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · 
Streamlit · yfinance

## How to Run Locally

pip install -r requirements.txt
python -m streamlit run app.py
