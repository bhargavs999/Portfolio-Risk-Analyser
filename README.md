# 📊 Portfolio Risk Analyser

> A data-driven portfolio analysis tool for Indian equity investors — built with Python and Streamlit. Select your NSE stocks, set your investment amount, and get a complete picture of optimal allocation, historical performance, risk metrics, and future simulations powered by real market data.

---

## 🚀 Live Demo

**👉 [Click here to open the app](<YOUR_STREAMLIT_LINK_HERE>)**

---

## 📌 What It Does

The Portfolio Risk Analyser applies Modern Portfolio Theory (MPT) to real NSE stock data, helping retail investors make evidence-based allocation decisions. It combines the Efficient Frontier, Monte Carlo simulation, historical backtesting, and Value at Risk — the same frameworks used by professional fund managers and reported by banks to regulators daily.

---

## ✨ Features

### 1. 📈 Current Market Data
- Live snapshot of each selected stock via Yahoo Finance
- Current price, 52-week high/low, daily change
- Position indicator — Near High / Mid Range / Near Low

### 2. 🎯 Efficient Frontier & Optimal Portfolio
- Simulates 10,000 random weight combinations
- Plots the full Efficient Frontier coloured by Sharpe Ratio
- Identifies the **Maximum Sharpe (Optimal)** and **Minimum Volatility (Safest)** portfolios

### 3. 💰 Optimal Investment Allocation
- Exact percentage and rupee allocation per stock
- Based on Maximum Sharpe Ratio optimisation
- Derived from 3 years of real NSE historical data

### 4. 🥧 Sector Allocation Breakdown
- Pie chart of sector exposure in your optimal portfolio
- Identifies dominant sector and diversification quality

### 5. 🔁 Historical Backtesting (2022–2024)
- Applies optimal weights to 3 years of real price data
- Tracks your portfolio's rupee value every trading day
- Shows profit/loss zones visually

### 6. 📊 Benchmark Comparison vs NIFTY 50
- Compares your portfolio directly against India's benchmark index
- Highlights periods of outperformance and underperformance

### 7. 🏆 Portfolio vs Individual Stocks
- Proves whether diversification beats stock-picking
- Horizontal bar chart comparing each stock's return vs the optimal portfolio

### 8. 📉 Maximum Drawdown Analysis
- Shows the biggest peak-to-trough fall your portfolio experienced
- Identifies the worst date and recovery status

### 9. 📊 Daily Return Distributions
- Histograms of daily returns for each stock
- Visualises volatility and return skew per stock

### 10. 🔗 Correlation Matrix
- Heatmap of pairwise correlations across all selected stocks
- Identifies the least correlated pair for strongest diversification

### 11. 🎲 Monte Carlo Simulation — Value at Risk
- Simulates up to 2,000 future portfolio scenarios
- Calculates VaR 95%, VaR 99%, CVaR (Expected Shortfall), and Median Outcome
- Distribution histogram of all simulated outcomes

### 12. 📋 Complete Summary
- Full written investment brief covering allocation, performance, risk, and key takeaways

---

## 📐 Financial Concepts Used

| Concept | What It Means |
|--------|---------------|
| **Sharpe Ratio** | Return earned per unit of risk — higher is better |
| **Efficient Frontier** | The set of portfolios that maximise return for a given risk level |
| **Maximum Drawdown** | Largest peak-to-trough loss in the portfolio's history |
| **VaR 95%** | 95% probability your loss won't exceed this value |
| **VaR 99%** | 99% probability your loss won't exceed this value |
| **CVaR / Expected Shortfall** | Average loss across the worst 5% of scenarios |
| **FOIR** | Fixed Obligation to Income Ratio — used by banks for loan approval |
| **Correlation** | How two stocks move relative to each other (−1 to +1) |

---

## 🏦 Stocks Available

20 major NSE-listed stocks across 8 sectors:

| Sector | Stocks |
|--------|--------|
| IT | TCS, Infosys, Wipro |
| Banking | HDFC Bank, ICICI Bank, SBI |
| FMCG | HUL, ITC, Nestle |
| Energy | Reliance, ONGC |
| Auto | Maruti, Bajaj Auto |
| Pharma | Sun Pharma, Dr Reddy |
| Tech | Zomato, Paytm |
| Infra / Consumer | L&T, Adani Ports, Asian Paints |

---

## ⚙️ Setup & Run Locally

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/bhargavs999/portfolio-risk-analyser.git
cd portfolio-risk-analyser
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — no environment variables needed

---

## 📦 Requirements

```
streamlit
numpy
pandas
matplotlib
seaborn
yfinance
```

---

## 🗂 Project Structure

```
portfolio-risk-analyser/
├── app.py              # Full application — all sections in one file
├── requirements.txt
└── README.md
```

---

## 💡 Key Design Decisions

**Single-file architecture** — The entire app lives in `app.py` for simplicity and ease of deployment. No database, no backend, no API keys required.

**Live data via yfinance** — All stock prices are fetched directly from Yahoo Finance (NSE tickers with `.NS` suffix), ensuring the analysis is always based on real market data.

**10,000 frontier simulations** — The Efficient Frontier uses 10,000 random Dirichlet-sampled weight vectors to ensure dense, accurate coverage of the feasible portfolio space.

**Institutional-grade risk metrics** — VaR and CVaR are the exact measures banks like JPMorgan Chase report to regulators. Including them makes this tool genuinely educational for finance students and professionals.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It does not constitute financial advice. Past performance of any stock or portfolio does not guarantee future results. Always consult a SEBI-registered financial advisor before making investment decisions.

---

## 👤 Author

**Bhargav Singh**  
D.J. Sanghvi College of Engineering, Mumbai · 2026  
[LinkedIn](https://www.linkedin.com/in/bhargavsingh9) · [GitHub](https://github.com/bhargavs999)

---

*Data source: Yahoo Finance / NSE. Not affiliated with NSE, BSE, or SEBI.*
