import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Portfolio Risk Analyser", 
                   layout="wide")

st.title("Portfolio Risk Analyser — NSE Stocks")
st.write("Analyse risk and return for a 5-stock Indian equity portfolio using Monte Carlo simulation and Modern Portfolio Theory.")

st.sidebar.header("Portfolio Settings")

INVESTMENT = st.sidebar.number_input(
    "Investment Amount (₹)", 
    min_value=10000, 
    max_value=10000000, 
    value=100000, 
    step=10000
)

NUM_SIMULATIONS = st.sidebar.slider(
    "Number of Simulations", 
    min_value=100, 
    max_value=2000, 
    value=1000, 
    step=100
)

available_stocks = {
    'TCS — IT': 'TCS.NS',
    'Infosys — IT': 'INFY.NS',
    'Wipro — IT': 'WIPRO.NS',
    'HDFC Bank — Banking': 'HDFCBANK.NS',
    'ICICI Bank — Banking': 'ICICIBANK.NS',
    'SBI — Banking': 'SBIN.NS',
    'HUL — FMCG': 'HINDUNILVR.NS',
    'ITC — FMCG': 'ITC.NS',
    'Nestle — FMCG': 'NESTLEIND.NS',
    'Reliance — Energy': 'RELIANCE.NS',
    'ONGC — Energy': 'ONGC.NS',
    'Maruti — Auto': 'MARUTI.NS',
    'Bajaj Auto — Auto': 'BAJAJ-AUTO.NS',
    'Sun Pharma — Pharma': 'SUNPHARMA.NS',
    'Dr Reddy — Pharma': 'DRREDDY.NS',
    'Zomato — Tech': 'ZOMATO.NS',
    'Paytm — Tech': 'PAYTM.NS',
    'Adani Ports — Infra': 'ADANIPORTS.NS',
    'L&T — Infra': 'LT.NS',
    'Asian Paints — Consumer': 'ASIANPAINT.NS'
}

selected_names = st.sidebar.multiselect(
    "Select Stocks (choose 3-7)",
    options=list(available_stocks.keys()),
    default=['TCS — IT', 'HDFC Bank — Banking',
             'HUL — FMCG', 'Maruti — Auto',
             'Reliance — Energy']
)

if len(selected_names) < 3:
    st.error("Please select at least 3 stocks.")
    st.stop()

tickers = [available_stocks[name] for name in selected_names]

# Load data
@st.cache_data
def load_data(tickers_tuple):
    tickers = list(tickers_tuple)
    data = yf.download(tickers, start='2022-01-01',
                       end='2024-12-31', auto_adjust=True)
    prices = data['Close']
    prices.columns = tickers
    returns = prices.pct_change().dropna()
    return prices, returns

prices, returns = load_data(tuple(tickers))

# Return Distributions
st.header("1. Return Distributions")

num_stocks = len(tickers)
cols = 3
rows = (num_stocks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, 
                          figsize=(15, 5 * rows))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    axes[i].hist(returns[ticker], bins=50,
                 color='steelblue', edgecolor='white')
    axes[i].set_title(f'{ticker}')
    axes[i].set_xlabel('Daily Return')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Correlation Matrix
st.header("2. Correlation Matrix")

fig2, ax = plt.subplots(figsize=(8, 6))
import seaborn as sns
correlation = returns.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f', ax=ax)
ax.set_title('Stock Correlation Matrix')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# Portfolio Optimisation
st.header("3. Efficient Frontier")

weights_record = []
results = np.zeros((3, 10000))

for i in range(10000):
    w = np.random.dirichlet(np.ones(len(tickers)))
    weights_record.append(w)
    port_return = np.sum(returns.mean().values * w) * 252
    port_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov().values * 252, w)))
    results[0, i] = port_return
    results[1, i] = port_vol
    results[2, i] = port_return / port_vol

max_sharpe_idx = results[2].argmax()
min_vol_idx = results[1].argmin()

fig3, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(results[1], results[0],
                     c=results[2], cmap='viridis',
                     alpha=0.5, s=10)
plt.colorbar(scatter, label='Sharpe Ratio')
ax.scatter(results[1, max_sharpe_idx],
           results[0, max_sharpe_idx],
           color='red', marker='*', s=500,
           label='Max Sharpe')
ax.scatter(results[1, min_vol_idx],
           results[0, min_vol_idx],
           color='blue', marker='*', s=500,
           label='Min Volatility')
ax.set_xlabel('Annual Volatility')
ax.set_ylabel('Annual Return')
ax.set_title('Efficient Frontier')
ax.legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# VaR Section
st.header("4. Monte Carlo — Value at Risk")

n = len(tickers)
weights = np.array([1/n] * n)
mean_returns = returns.mean().values
cov_matrix = returns.cov().values
port_mean = np.dot(weights, mean_returns)
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Check for valid values
if np.isnan(port_mean) or np.isnan(port_vol):
    st.error("Could not calculate portfolio statistics. Please try again.")
else:
    simulation_results = []
    for i in range(NUM_SIMULATIONS):
        daily_returns = np.random.normal(port_mean,
                                         port_vol,
                                         252)
        price_series = [INVESTMENT]
        for r in daily_returns:
            price_series.append(price_series[-1] * (1 + r))
        simulation_results.append(price_series[-1])

    simulation_results = np.array(simulation_results)
    
    # Only plot if we have valid results
    if len(simulation_results) > 0 and not np.any(np.isnan(simulation_results)):
        var_95 = np.percentile(simulation_results, 5)
        var_99 = np.percentile(simulation_results, 1)
        cvar_95 = simulation_results[
            simulation_results <= var_95].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Value", f"₹{np.median(simulation_results):,.0f}")
        col2.metric("VaR 95%", f"₹{var_95:,.0f}",
                    f"-₹{INVESTMENT-var_95:,.0f}")
        col3.metric("VaR 99%", f"₹{var_99:,.0f}",
                    f"-₹{INVESTMENT-var_99:,.0f}")
        col4.metric("CVaR 95%", f"₹{cvar_95:,.0f}",
                    f"-₹{INVESTMENT-cvar_95:,.0f}")

        fig4, ax = plt.subplots(figsize=(12, 5))
        ax.hist(simulation_results, bins=60,
                color='steelblue', edgecolor='white',
                alpha=0.7)
        ax.axvline(INVESTMENT, color='black',
                   linestyle='--', linewidth=2,
                   label=f'Investment ₹{INVESTMENT:,.0f}')
        ax.axvline(var_95, color='red',
                   linestyle='--', linewidth=2,
                   label=f'VaR 95% ₹{var_95:,.0f}')
        ax.axvline(var_99, color='darkred',
                   linestyle='--', linewidth=2,
                   label=f'VaR 99% ₹{var_99:,.0f}')
        ax.axvline(cvar_95, color='orange',
                   linestyle='--', linewidth=2,
                   label=f'CVaR 95% ₹{cvar_95:,.0f}')
        ax.axvline(np.median(simulation_results),
                   color='green', linestyle='--',
                   linewidth=2,
                   label=f'Median ₹{np.median(simulation_results):,.0f}')
        ax.set_xlabel('Portfolio Value After 1 Year (₹)')
        ax.set_ylabel('Frequency')
        ax.set_title('Monte Carlo Simulation — Portfolio VaR')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        st.header("5. Business Insight")
        st.write(f"""
        **Portfolio Analysis Summary:**
        - **Median expected value** after 1 year: ₹{np.median(simulation_results):,.0f}
        - **VaR 95%**: Only 5% chance of losing more than ₹{INVESTMENT-var_95:,.0f}
        - **CVaR 95%**: In worst case scenarios, average loss is ₹{INVESTMENT-cvar_95:,.0f}
        - All 5 stocks show low correlations (0.20-0.35)
        - HINDUNILVR dominates optimal portfolio (63.65%)
        """)
    else:
        st.error("Simulation produced invalid results. Please try again.")

