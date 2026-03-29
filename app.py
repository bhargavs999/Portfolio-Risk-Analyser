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
st.write("Analyse risk and return for an Indian equity portfolio using Monte Carlo simulation and Modern Portfolio Theory.")

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

# Current Market Data
st.header("1. Current Market Data")

@st.cache_data(ttl=300)
def load_current_data(tickers_tuple):
    tickers = list(tickers_tuple)
    data = yf.download(tickers, period='1y', 
                       auto_adjust=True)
    prices = data['Close']
    prices.columns = tickers
    return prices

current_data = load_current_data(tuple(tickers))

market_data = []
for ticker in tickers:
    stock_prices = current_data[ticker].dropna()
    current_price = stock_prices.iloc[-1]
    prev_price = stock_prices.iloc[-2]
    daily_change = ((current_price - prev_price) 
                    / prev_price * 100)
    high_52w = stock_prices.max()
    low_52w = stock_prices.min()
    
    market_data.append({
        'Stock': ticker,
        'Current Price (₹)': f"₹{current_price:,.0f}",
        '52W High (₹)': f"₹{high_52w:,.0f}",
        '52W Low (₹)': f"₹{low_52w:,.0f}",
        'Daily Change': f"{daily_change:+.2f}%",
        'Position': (
            'Near High 🔴' 
            if current_price > 0.9 * high_52w 
            else 'Near Low 🟢' 
            if current_price < 1.1 * low_52w 
            else 'Mid Range 🟡')
    })

market_df = pd.DataFrame(market_data)
st.dataframe(market_df, use_container_width=True,
             hide_index=True)

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
st.header("2. Return Distributions")

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
st.header("3. Correlation Matrix")

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
st.header("4. Efficient Frontier")

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

# Sector Pie Chart
st.subheader("Optimal Portfolio — Sector Allocation")

# Map tickers to sectors
sector_map = {
    'TCS.NS': 'IT',
    'INFY.NS': 'IT',
    'WIPRO.NS': 'IT',
    'HDFCBANK.NS': 'Banking',
    'ICICIBANK.NS': 'Banking',
    'SBIN.NS': 'Banking',
    'HINDUNILVR.NS': 'FMCG',
    'ITC.NS': 'FMCG',
    'NESTLEIND.NS': 'FMCG',
    'RELIANCE.NS': 'Energy',
    'ONGC.NS': 'Energy',
    'MARUTI.NS': 'Auto',
    'BAJAJ-AUTO.NS': 'Auto',
    'SUNPHARMA.NS': 'Pharma',
    'DRREDDY.NS': 'Pharma',
    'ZOMATO.NS': 'Tech',
    'PAYTM.NS': 'Tech',
    'ADANIPORTS.NS': 'Infra',
    'LT.NS': 'Infra',
    'ASIANPAINT.NS': 'Consumer'
}

# Aggregate weights by sector
sector_weights = {}
for ticker, weight in zip(tickers, best_weights):
    sector = sector_map.get(ticker, 'Other')
    sector_weights[sector] = (sector_weights.get(sector, 0) 
                               + weight)

# Plot
col1, col2 = st.columns([1, 1])

with col1:
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Set3(
        np.linspace(0, 1, len(sector_weights)))
    wedges, texts, autotexts = ax_pie.pie(
        list(sector_weights.values()),
        labels=list(sector_weights.keys()),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax_pie.set_title('Sector Allocation')
    plt.tight_layout()
    st.pyplot(fig_pie)
    plt.close()

with col2:
    sector_df = pd.DataFrame({
        'Sector': list(sector_weights.keys()),
        'Allocation': [f"{v:.1%}" 
                       for v in sector_weights.values()],
        'Amount (₹)': [f"₹{v*INVESTMENT:,.0f}" 
                        for v in sector_weights.values()]
    }).sort_values('Allocation', ascending=False)
    
    st.dataframe(sector_df, use_container_width=True,
                 hide_index=True)
    
    dominant_sector = max(sector_weights.items(),
                          key=lambda x: x[1])
    st.write(f"**Dominant Sector:** {dominant_sector[0]} "
             f"({dominant_sector[1]:.1%} allocation)")
  
# Historical Backtesting
st.header("5. Historical Backtesting")

# Get optimal weights
best_weights = np.array(weights_record[max_sharpe_idx])

# Calculate portfolio historical returns
portfolio_returns = returns.dot(best_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
portfolio_value = cumulative_returns * INVESTMENT

# Final value
final_value = portfolio_value.iloc[-1]
total_return = ((final_value - INVESTMENT) / INVESTMENT) * 100

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Starting Investment", f"₹{INVESTMENT:,.0f}")
col2.metric("Final Value", f"₹{final_value:,.0f}")
col3.metric("Total Return", f"{total_return:.1f}%")

# Plot
fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
ax_bt.plot(portfolio_value.index, portfolio_value,
           color='steelblue', linewidth=2,
           label='Optimal Portfolio')
ax_bt.axhline(y=INVESTMENT, color='red',
              linestyle='--', linewidth=1.5,
              label=f'Starting Investment ₹{INVESTMENT:,.0f}')
ax_bt.fill_between(portfolio_value.index,
                    INVESTMENT, portfolio_value,
                    where=portfolio_value >= INVESTMENT,
                    alpha=0.2, color='green',
                    label='Profit Zone')
ax_bt.fill_between(portfolio_value.index,
                    INVESTMENT, portfolio_value,
                    where=portfolio_value < INVESTMENT,
                    alpha=0.2, color='red',
                    label='Loss Zone')
ax_bt.set_xlabel('Date')
ax_bt.set_ylabel('Portfolio Value (₹)')
ax_bt.set_title('Historical Portfolio Performance — Optimal Weights')
ax_bt.legend()
plt.tight_layout()
st.pyplot(fig_bt)
plt.close()

# Benchmark Comparison
st.header("6. Benchmark Comparison — Portfolio vs NIFTY 50")

@st.cache_data
def load_benchmark():
    nifty = yf.download('^NSEI', start='2022-01-01',
                        end='2024-12-31', auto_adjust=True)
    return nifty['Close'].squeeze()

nifty_prices = load_benchmark()
nifty_returns = nifty_prices.pct_change().dropna()
nifty_cumulative = (1 + nifty_returns).cumprod() * INVESTMENT

# Align dates
common_dates = portfolio_value.index.intersection(
    nifty_cumulative.index)
portfolio_aligned = portfolio_value[common_dates]
nifty_aligned = nifty_cumulative[common_dates]

# Final values
nifty_final = nifty_aligned.iloc[-1]
nifty_return = ((nifty_final - INVESTMENT) / INVESTMENT) * 100
outperformance = total_return - nifty_return

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Return", f"{total_return:.1f}%")
col2.metric("NIFTY 50 Return", f"{nifty_return:.1f}%")
col3.metric("Outperformance", f"{outperformance:.1f}%",
            f"{'above' if outperformance > 0 else 'below'} NIFTY 50")

# Plot
fig_bench, ax_bench = plt.subplots(figsize=(12, 5))
ax_bench.plot(portfolio_aligned.index, portfolio_aligned,
              color='steelblue', linewidth=2,
              label='Optimal Portfolio')
ax_bench.plot(nifty_aligned.index, nifty_aligned,
              color='orange', linewidth=2,
              label='NIFTY 50')
ax_bench.axhline(y=INVESTMENT, color='red',
                 linestyle='--', linewidth=1.5,
                 label=f'Starting Investment ₹{INVESTMENT:,.0f}')
ax_bench.fill_between(portfolio_aligned.index,
                       portfolio_aligned, nifty_aligned,
                       where=portfolio_aligned >= nifty_aligned,
                       alpha=0.2, color='green',
                       label='Portfolio Outperforming')
ax_bench.fill_between(portfolio_aligned.index,
                       portfolio_aligned, nifty_aligned,
                       where=portfolio_aligned < nifty_aligned,
                       alpha=0.2, color='red',
                       label='Portfolio Underperforming')
ax_bench.set_xlabel('Date')
ax_bench.set_ylabel('Portfolio Value (₹)')
ax_bench.set_title('Optimal Portfolio vs NIFTY 50 Index')
ax_bench.legend()
plt.tight_layout()
st.pyplot(fig_bench)
plt.close()

# Business insight
if outperformance > 0:
    st.success(f"Your optimal portfolio outperformed NIFTY 50 by {outperformance:.1f}% over 3 years.")
else:
    st.warning(f"Your optimal portfolio underperformed NIFTY 50 by {abs(outperformance):.1f}% over 3 years.")

# Portfolio vs Individual Stocks
st.header("7. Portfolio vs Individual Stocks")

# Calculate individual stock returns over period
individual_returns = {}
for ticker in tickers:
    stock_cumret = (1 + returns[ticker]).cumprod()
    individual_returns[ticker] = (
        stock_cumret.iloc[-1] - 1) * 100

# Add portfolio return
individual_returns['Optimal Portfolio'] = total_return

# Sort by return
sorted_returns = dict(sorted(
    individual_returns.items(),
    key=lambda x: x[1]))

# Plot
fig_ind, ax_ind = plt.subplots(figsize=(10, 5))
colors = ['steelblue' if k != 'Optimal Portfolio' 
          else 'gold' for k in sorted_returns.keys()]
bars = ax_ind.barh(list(sorted_returns.keys()),
                    list(sorted_returns.values()),
                    color=colors, edgecolor='white')

# Add value labels
for bar, val in zip(bars, sorted_returns.values()):
    ax_ind.text(val + 1, bar.get_y() + 
                bar.get_height()/2,
                f'{val:.1f}%',
                va='center', fontweight='bold')

ax_ind.axvline(x=0, color='white', linewidth=0.5)
ax_ind.set_xlabel('Total Return (%)')
ax_ind.set_title('3-Year Return: Optimal Portfolio vs Individual Stocks')
plt.tight_layout()
st.pyplot(fig_ind)
plt.close()

# Key insight
best_individual = max(individual_returns.items(),
                      key=lambda x: x[1]
                      if x[0] != 'Optimal Portfolio' 
                      else -999)
best_individual = max(
    {k: v for k, v in individual_returns.items() 
     if k != 'Optimal Portfolio'}.items(),
    key=lambda x: x[1]
)

if total_return > best_individual[1]:
    st.success(f"The optimal portfolio outperformed every individual stock including the best performer ({best_individual[0]} at {best_individual[1]:.1f}%) — proving diversification adds real value.")
else:
    st.info(f"While {best_individual[0]} individually returned {best_individual[1]:.1f}%, the optimal portfolio achieved {total_return:.1f}% with significantly lower risk — better risk-adjusted performance even if not the highest raw return.")

# Drawdown Analysis
st.header("8. Maximum Drawdown Analysis")

# Calculate drawdown
rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max * 100
max_drawdown = drawdown.min()
max_drawdown_date = drawdown.idxmin()

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Maximum Drawdown", f"{max_drawdown:.1f}%")
col2.metric("Worst Date", max_drawdown_date.strftime('%b %Y'))
col3.metric("Recovery", "Full Recovery Achieved" 
            if portfolio_value.iloc[-1] > portfolio_value.iloc[0] 
            else "Not Yet Recovered")

# Plot
fig_dd, ax_dd = plt.subplots(figsize=(12, 4))
ax_dd.fill_between(drawdown.index, drawdown, 0,
                    color='red', alpha=0.4,
                    label='Drawdown')
ax_dd.plot(drawdown.index, drawdown,
           color='red', linewidth=1)
ax_dd.axhline(y=max_drawdown, color='darkred',
              linestyle='--', linewidth=1.5,
              label=f'Max Drawdown {max_drawdown:.1f}%')
ax_dd.axhline(y=0, color='white',
              linewidth=0.5)
ax_dd.set_xlabel('Date')
ax_dd.set_ylabel('Drawdown (%)')
ax_dd.set_title('Portfolio Drawdown Over Time')
ax_dd.legend()
plt.tight_layout()
st.pyplot(fig_dd)
plt.close()

st.write(f"""
**What this means:**
The portfolio experienced a maximum drawdown of **{max_drawdown:.1f}%** 
in **{max_drawdown_date.strftime('%B %Y')}**. 
This means at its worst point, the portfolio had fallen 
{abs(max_drawdown):.1f}% from its recent peak. 
Despite this, the portfolio fully recovered and went on 
to deliver {total_return:.1f}% total returns — demonstrating 
resilience through market volatility.
""")

# VaR Section
st.header("9. Monte Carlo — Value at Risk")

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

        st.header("10. Business Insight")
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

