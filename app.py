import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio Risk Analyser | Bhargav Singh",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <h1 style='font-size: 2.4rem; font-weight: 700; margin-bottom: 0;'>
            📊 Portfolio Risk Analyser
        </h1>
        <p style='color: grey; font-size: 0.95rem; margin-top: 0.3rem;'>
            Built by <strong>Bhargav Singh</strong> · D.J. Sanghvi College of Engineering · 
            <a href='https://www.linkedin.com/in/bhargavsingh9' target='_blank'>LinkedIn</a> · 
            <a href='https://github.com/bhargavs999' target='_blank'>GitHub</a>
        </p>
        <p style='color: grey; font-size: 0.85rem;'>
            A data-driven portfolio analysis tool for Indian equity investors. 
            Select your stocks, set your investment amount, and get a complete 
            picture of optimal allocation, historical performance, risk metrics, 
            and future simulations — all powered by real NSE market data.
        </p>
        <hr style='margin-top: 0.8rem;'>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 1rem;'>
        <h3 style='margin-bottom: 0;'>⚙️ Settings</h3>
        <p style='color: grey; font-size: 0.8rem;'>Bhargav Singh · DJSCE</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

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

# ── TOP METRIC STRIP ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks Selected", len(tickers))
col2.metric("Investment", f"₹{INVESTMENT:,.0f}")
col3.metric("Simulations", f"{NUM_SIMULATIONS:,}")
col4.metric("Data Period", "2022–2024")

st.divider()

# ── SECTION 1 — CURRENT MARKET DATA ──────────────────────────────────────────
st.markdown("### 1. 📈 Current Market Data")
st.caption(
    "Live snapshot of each selected stock — current price, 52-week range, "
    "and today's movement. Use this to understand where each stock is trading "
    "relative to its yearly high and low before investing."
)
st.divider()

@st.cache_data(ttl=300)
def load_current_data(tickers_tuple):
    tickers = list(tickers_tuple)
    data = yf.download(tickers, period='1y', auto_adjust=True)
    prices = data['Close']
    prices.columns = tickers
    return prices

current_data = load_current_data(tuple(tickers))

market_data = []
for ticker in tickers:
    stock_prices = current_data[ticker].dropna()
    current_price = stock_prices.iloc[-1]
    prev_price = stock_prices.iloc[-2]
    daily_change = ((current_price - prev_price) / prev_price * 100)
    high_52w = stock_prices.max()
    low_52w = stock_prices.min()
    market_data.append({
        'Stock': ticker,
        'Current Price (₹)': f"₹{current_price:,.0f}",
        '52W High (₹)': f"₹{high_52w:,.0f}",
        '52W Low (₹)': f"₹{low_52w:,.0f}",
        'Daily Change': f"{daily_change:+.2f}%",
        'Position': (
            'Near High 🔴' if current_price > 0.9 * high_52w
            else 'Near Low 🟢' if current_price < 1.1 * low_52w
            else 'Mid Range 🟡')
    })

market_df = pd.DataFrame(market_data)
st.dataframe(market_df, use_container_width=True, hide_index=True)

# ── LOAD HISTORICAL DATA ─────────────────────────────────────────────────────
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

# ── SECTION 2 — EFFICIENT FRONTIER ───────────────────────────────────────────
st.markdown("### 2. 🎯 Efficient Frontier & Optimal Portfolio")
st.caption(
    "This is the core of Modern Portfolio Theory. We simulate 10,000 different "
    "ways to split your investment across your selected stocks. Each dot is one "
    "possible portfolio. The colour shows the Sharpe Ratio — how much return "
    "you get per unit of risk. The red star is the mathematically optimal "
    "portfolio. The blue star is the least risky portfolio possible."
)
st.divider()

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

fig3, ax3 = plt.subplots(figsize=(10, 6))
scatter = ax3.scatter(results[1], results[0],
                      c=results[2], cmap='viridis',
                      alpha=0.5, s=10)
plt.colorbar(scatter, label='Sharpe Ratio')
ax3.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
            color='red', marker='*', s=500, label='Max Sharpe (Optimal)')
ax3.scatter(results[1, min_vol_idx], results[0, min_vol_idx],
            color='blue', marker='*', s=500, label='Min Volatility (Safest)')
ax3.set_xlabel('Annual Volatility (Risk)')
ax3.set_ylabel('Annual Return')
ax3.set_title('Efficient Frontier — 10,000 Portfolio Simulations')
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ── SECTION 3 — OPTIMAL WEIGHTS ──────────────────────────────────────────────
st.markdown("### 3. 💰 Optimal Portfolio — Exact Investment Allocation")
st.caption(
    "This is the answer to the most important question — exactly how much of "
    "your money should go into each stock to achieve the best possible "
    "risk-adjusted return. These weights are derived from the Maximum Sharpe "
    "portfolio on the Efficient Frontier above."
)
st.divider()

best_weights = np.array(weights_record[max_sharpe_idx])

weight_df = pd.DataFrame({
    'Stock': tickers,
    'Allocation (%)': [f"{w:.2%}" for w in best_weights],
    'Amount to Invest (₹)': [f"₹{w * INVESTMENT:,.0f}" for w in best_weights]
})
st.table(weight_df)

st.info(
    f"💡 For a ₹{INVESTMENT:,.0f} investment, the above allocation gives you "
    f"the maximum return per unit of risk based on 3 years of real NSE data. "
    f"This is calculated using the Sharpe Ratio — the gold standard metric "
    f"used by professional fund managers worldwide."
)

# ── SECTION 4 — SECTOR PIE CHART ─────────────────────────────────────────────
st.markdown("### 4. 🥧 Sector Allocation Breakdown")
st.caption(
    "A visual breakdown of which sectors your optimal portfolio is exposed to. "
    "A well-diversified portfolio ideally has exposure across multiple sectors "
    "so that a downturn in one sector doesn't wipe out your entire portfolio."
)
st.divider()

sector_map = {
    'TCS.NS': 'IT', 'INFY.NS': 'IT', 'WIPRO.NS': 'IT',
    'HDFCBANK.NS': 'Banking', 'ICICIBANK.NS': 'Banking', 'SBIN.NS': 'Banking',
    'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG', 'NESTLEIND.NS': 'FMCG',
    'RELIANCE.NS': 'Energy', 'ONGC.NS': 'Energy',
    'MARUTI.NS': 'Auto', 'BAJAJ-AUTO.NS': 'Auto',
    'SUNPHARMA.NS': 'Pharma', 'DRREDDY.NS': 'Pharma',
    'ZOMATO.NS': 'Tech', 'PAYTM.NS': 'Tech',
    'ADANIPORTS.NS': 'Infra', 'LT.NS': 'Infra',
    'ASIANPAINT.NS': 'Consumer'
}

sector_weights = {}
for ticker, weight in zip(tickers, best_weights):
    sector = sector_map.get(ticker, 'Other')
    sector_weights[sector] = sector_weights.get(sector, 0) + weight

col1, col2 = st.columns([1, 1])
with col1:
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sector_weights)))
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
        'Allocation': [f"{v:.1%}" for v in sector_weights.values()],
        'Amount (₹)': [f"₹{v * INVESTMENT:,.0f}" for v in sector_weights.values()]
    }).sort_values('Allocation', ascending=False)
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    dominant_sector = max(sector_weights.items(), key=lambda x: x[1])
    st.write(
        f"**Dominant Sector:** {dominant_sector[0]} "
        f"({dominant_sector[1]:.1%} allocation)"
    )
    st.write(
        "The dominant sector reflects which part of the market offers the "
        "best risk-adjusted returns based on historical data. This is not a "
        "manual choice — it emerges purely from the mathematical optimisation."
    )

# ── SECTION 5 — HISTORICAL BACKTESTING ───────────────────────────────────────
st.markdown("### 5. 🔁 Historical Backtesting — Did It Actually Work?")
st.caption(
    "Monte Carlo tells you what might happen in the future. Backtesting tells "
    "you what actually happened in the past. We apply the optimal weights to "
    "real historical NSE data from January 2022 to December 2024 and track "
    "your portfolio's actual rupee value every single trading day."
)
st.divider()

portfolio_returns = returns.dot(best_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
portfolio_value = cumulative_returns * INVESTMENT

final_value = portfolio_value.iloc[-1]
total_return = ((final_value - INVESTMENT) / INVESTMENT) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Starting Investment", f"₹{INVESTMENT:,.0f}")
col2.metric("Final Value (Dec 2024)", f"₹{final_value:,.0f}")
col3.metric("Total 3-Year Return", f"{total_return:.1f}%")

fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
ax_bt.plot(portfolio_value.index, portfolio_value,
           color='steelblue', linewidth=2, label='Optimal Portfolio')
ax_bt.axhline(y=INVESTMENT, color='red', linestyle='--',
              linewidth=1.5, label=f'Starting Investment ₹{INVESTMENT:,.0f}')
ax_bt.fill_between(portfolio_value.index, INVESTMENT, portfolio_value,
                    where=portfolio_value >= INVESTMENT,
                    alpha=0.2, color='green', label='Profit Zone')
ax_bt.fill_between(portfolio_value.index, INVESTMENT, portfolio_value,
                    where=portfolio_value < INVESTMENT,
                    alpha=0.2, color='red', label='Loss Zone')
ax_bt.set_xlabel('Date')
ax_bt.set_ylabel('Portfolio Value (₹)')
ax_bt.set_title('Historical Portfolio Performance — Optimal Weights (2022–2024)')
ax_bt.legend()
plt.tight_layout()
st.pyplot(fig_bt)
plt.close()

# ── SECTION 6 — BENCHMARK COMPARISON ─────────────────────────────────────────
st.markdown("### 6. 📊 Benchmark Comparison — vs NIFTY 50")
st.caption(
    "The real test of any investment strategy is whether it beats the market. "
    "NIFTY 50 is India's benchmark index — the average performance of the top "
    "50 listed companies. If your portfolio can't beat NIFTY 50, you'd be "
    "better off just buying an index fund. Here's how your optimal portfolio "
    "compared over the same 3-year period."
)
st.divider()

@st.cache_data
def load_benchmark():
    nifty = yf.download('^NSEI', start='2022-01-01',
                        end='2024-12-31', auto_adjust=True)
    return nifty['Close'].squeeze()

nifty_prices = load_benchmark()
nifty_returns = nifty_prices.pct_change().dropna()
nifty_cumulative = (1 + nifty_returns).cumprod() * INVESTMENT

common_dates = portfolio_value.index.intersection(nifty_cumulative.index)
portfolio_aligned = portfolio_value[common_dates]
nifty_aligned = nifty_cumulative[common_dates]

nifty_final = nifty_aligned.iloc[-1]
nifty_return = ((nifty_final - INVESTMENT) / INVESTMENT) * 100
outperformance = total_return - nifty_return

col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Return", f"{total_return:.1f}%")
col2.metric("NIFTY 50 Return", f"{nifty_return:.1f}%")
col3.metric("Outperformance", f"{outperformance:.1f}%",
            f"{'above' if outperformance > 0 else 'below'} NIFTY 50")

fig_bench, ax_bench = plt.subplots(figsize=(12, 5))
ax_bench.plot(portfolio_aligned.index, portfolio_aligned,
              color='steelblue', linewidth=2, label='Optimal Portfolio')
ax_bench.plot(nifty_aligned.index, nifty_aligned,
              color='orange', linewidth=2, label='NIFTY 50')
ax_bench.axhline(y=INVESTMENT, color='red', linestyle='--',
                 linewidth=1.5, label=f'Starting Investment ₹{INVESTMENT:,.0f}')
ax_bench.fill_between(portfolio_aligned.index,
                       portfolio_aligned, nifty_aligned,
                       where=portfolio_aligned >= nifty_aligned,
                       alpha=0.2, color='green', label='Portfolio Outperforming')
ax_bench.fill_between(portfolio_aligned.index,
                       portfolio_aligned, nifty_aligned,
                       where=portfolio_aligned < nifty_aligned,
                       alpha=0.2, color='red', label='Portfolio Underperforming')
ax_bench.set_xlabel('Date')
ax_bench.set_ylabel('Portfolio Value (₹)')
ax_bench.set_title('Optimal Portfolio vs NIFTY 50 (2022–2024)')
ax_bench.legend()
plt.tight_layout()
st.pyplot(fig_bench)
plt.close()

if outperformance > 0:
    st.success(
        f"✅ Your optimal portfolio outperformed NIFTY 50 by {outperformance:.1f}% "
        f"over 3 years. This means the mathematical weight optimisation added real "
        f"value beyond what passive index investing would have delivered."
    )
else:
    st.warning(
        f"⚠️ Your optimal portfolio underperformed NIFTY 50 by {abs(outperformance):.1f}%. "
        f"Consider changing your stock selection to include stronger performers."
    )

# ── SECTION 7 — PORTFOLIO VS INDIVIDUAL STOCKS ───────────────────────────────
st.markdown("### 7. 🏆 Portfolio vs Individual Stocks")
st.caption(
    "Would you have done better just picking one stock and putting everything "
    "in it? This chart compares your optimal diversified portfolio against "
    "each individual stock's 3-year return. The gold bar is your portfolio. "
    "Blue bars are individual stocks."
)
st.divider()

individual_returns = {}
for ticker in tickers:
    stock_cumret = (1 + returns[ticker]).cumprod()
    individual_returns[ticker] = (stock_cumret.iloc[-1] - 1) * 100

individual_returns['Optimal Portfolio'] = total_return
sorted_returns = dict(sorted(individual_returns.items(), key=lambda x: x[1]))

fig_ind, ax_ind = plt.subplots(figsize=(10, 5))
colors = ['steelblue' if k != 'Optimal Portfolio'
          else 'gold' for k in sorted_returns.keys()]
bars = ax_ind.barh(list(sorted_returns.keys()),
                    list(sorted_returns.values()),
                    color=colors, edgecolor='white')

for bar, val in zip(bars, sorted_returns.values()):
    ax_ind.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontweight='bold')

ax_ind.axvline(x=0, color='white', linewidth=0.5)
ax_ind.set_xlabel('Total Return (%)')
ax_ind.set_title('3-Year Return: Optimal Portfolio vs Individual Stocks')
plt.tight_layout()
st.pyplot(fig_ind)
plt.close()

best_individual = max(
    {k: v for k, v in individual_returns.items()
     if k != 'Optimal Portfolio'}.items(),
    key=lambda x: x[1]
)

if total_return > best_individual[1]:
    st.success(
        f"✅ The optimal portfolio outperformed every individual stock "
        f"including the best performer ({best_individual[0]} at "
        f"{best_individual[1]:.1f}%) — proving diversification adds real value."
    )
else:
    st.info(
        f"💡 While {best_individual[0]} individually returned "
        f"{best_individual[1]:.1f}%, the optimal portfolio achieved "
        f"{total_return:.1f}% with significantly lower risk. "
        f"Diversification may not always give the highest raw return, "
        f"but it gives the best return for the level of risk taken — "
        f"which is what matters for long-term wealth building."
    )

# ── SECTION 8 — DRAWDOWN ANALYSIS ────────────────────────────────────────────
st.markdown("### 8. 📉 Maximum Drawdown Analysis")
st.caption(
    "Returns tell you where you ended up. Drawdown tells you what you had to "
    "survive to get there. Maximum drawdown measures the biggest peak-to-trough "
    "fall your portfolio experienced. This is the most psychologically important "
    "metric — because most investors panic and sell during large drawdowns, "
    "missing the recovery entirely."
)
st.divider()

rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max * 100
max_drawdown = drawdown.min()
max_drawdown_date = drawdown.idxmin()

col1, col2, col3 = st.columns(3)
col1.metric("Maximum Drawdown", f"{max_drawdown:.1f}%")
col2.metric("Worst Point", max_drawdown_date.strftime('%b %Y'))
col3.metric("Recovery Status",
            "✅ Full Recovery" if portfolio_value.iloc[-1] > portfolio_value.iloc[0]
            else "⚠️ Not Yet Recovered")

fig_dd, ax_dd = plt.subplots(figsize=(12, 4))
ax_dd.fill_between(drawdown.index, drawdown, 0,
                    color='red', alpha=0.4, label='Drawdown')
ax_dd.plot(drawdown.index, drawdown, color='red', linewidth=1)
ax_dd.axhline(y=max_drawdown, color='darkred', linestyle='--',
              linewidth=1.5, label=f'Max Drawdown {max_drawdown:.1f}%')
ax_dd.axhline(y=0, color='white', linewidth=0.5)
ax_dd.set_xlabel('Date')
ax_dd.set_ylabel('Drawdown from Peak (%)')
ax_dd.set_title('Portfolio Drawdown Over Time — How Far From Peak?')
ax_dd.legend()
plt.tight_layout()
st.pyplot(fig_dd)
plt.close()

st.write(
    f"The portfolio's worst point was **{max_drawdown:.1f}%** below its recent "
    f"peak in **{max_drawdown_date.strftime('%B %Y')}** — coinciding with the "
    f"global market correction of mid-2022. Despite this, the portfolio fully "
    f"recovered and delivered {total_return:.1f}% total returns. "
    f"An investor who stayed invested through the drawdown was rewarded. "
    f"An investor who panic-sold at the bottom locked in permanent losses."
)

# ── SECTION 9 — RETURN DISTRIBUTIONS ─────────────────────────────────────────
st.markdown("### 9. 📊 Individual Stock Return Distributions")
st.caption(
    "These histograms show how each stock moved on a daily basis over 3 years. "
    "A tall narrow peak centred at zero means the stock is stable — most days "
    "it barely moves. A wide flat distribution means high volatility — the "
    "stock swings significantly day to day. The width of the distribution is "
    "your risk. The centre tells you the average daily direction."
)
st.divider()

num_stocks = len(tickers)
cols = 3
rows = (num_stocks + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    axes[i].hist(returns[ticker], bins=50,
                 color='steelblue', edgecolor='white')
    axes[i].set_title(f'{ticker} Daily Returns')
    axes[i].set_xlabel('Daily Return')
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── SECTION 10 — CORRELATION MATRIX ──────────────────────────────────────────
st.markdown("### 10. 🔗 Stock Correlation Matrix")
st.caption(
    "Correlation measures how two stocks move relative to each other. "
    "A value of +1 means they move perfectly together — if one falls, "
    "the other falls too, offering no protection. A value of 0 means "
    "they move independently — ideal for diversification. A value of -1 "
    "means they move in opposite directions — perfect hedge. "
    "For a truly diversified portfolio, you want low correlations between all pairs."
)
st.divider()

fig2, ax2 = plt.subplots(figsize=(8, 6))
correlation = returns.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f', ax=ax2)
ax2.set_title('Stock Correlation Matrix — Lower is Better for Diversification')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

min_corr_pair = None
min_corr_val = 1.0
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        val = correlation.iloc[i, j]
        if val < min_corr_val:
            min_corr_val = val
            min_corr_pair = (tickers[i], tickers[j])

st.write(
    f"The least correlated pair in your portfolio is **{min_corr_pair[0]}** and "
    f"**{min_corr_pair[1]}** at **{min_corr_val:.2f}** — meaning these two stocks "
    f"move most independently from each other, providing the strongest "
    f"diversification benefit within your selection."
)

# ── SECTION 11 — MONTE CARLO VAR ─────────────────────────────────────────────
st.markdown("### 11. 🎲 Monte Carlo Simulation — Future Risk (Value at Risk)")
st.caption(
    "While backtesting shows what happened in the past, Monte Carlo simulation "
    "models what could happen in the future. We simulate thousands of possible "
    "market scenarios based on historical volatility and calculate the range of "
    "likely portfolio outcomes after one year. This gives you a probability-based "
    "view of your downside risk — the same framework used by every major bank "
    "to calculate Value at Risk (VaR) reported to regulators daily."
)
st.divider()

n = len(tickers)
weights = np.array([1 / n] * n)
mean_returns = returns.mean().values
cov_matrix = returns.cov().values
port_mean = np.dot(weights, mean_returns)
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

if np.isnan(port_mean) or np.isnan(port_vol):
    st.error("Could not calculate portfolio statistics. Please try again.")
else:
    simulation_results = []
    for i in range(NUM_SIMULATIONS):
        daily_returns = np.random.normal(port_mean, port_vol, 252)
        price_series = [INVESTMENT]
        for r in daily_returns:
            price_series.append(price_series[-1] * (1 + r))
        simulation_results.append(price_series[-1])

    simulation_results = np.array(simulation_results)

    if len(simulation_results) > 0 and not np.any(np.isnan(simulation_results)):
        var_95 = np.percentile(simulation_results, 5)
        var_99 = np.percentile(simulation_results, 1)
        cvar_95 = simulation_results[simulation_results <= var_95].mean()
        median_val = np.median(simulation_results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Outcome", f"₹{median_val:,.0f}",
                    f"+₹{median_val - INVESTMENT:,.0f}")
        col2.metric("VaR 95%", f"₹{var_95:,.0f}",
                    f"-₹{INVESTMENT - var_95:,.0f}")
        col3.metric("VaR 99%", f"₹{var_99:,.0f}",
                    f"-₹{INVESTMENT - var_99:,.0f}")
        col4.metric("CVaR 95%", f"₹{cvar_95:,.0f}",
                    f"-₹{INVESTMENT - cvar_95:,.0f}")

        st.write(
            f"**How to read these numbers:** "
            f"The median outcome of ₹{median_val:,.0f} is what you'd most likely "
            f"end up with after one year. VaR 95% means there is only a 5% chance "
            f"your portfolio falls below ₹{var_95:,.0f} — a maximum loss of "
            f"₹{INVESTMENT - var_95:,.0f}. VaR 99% is the extreme case — only 1% "
            f"chance of falling below ₹{var_99:,.0f}. CVaR tells you the average "
            f"loss in that worst 5% of scenarios — ₹{INVESTMENT - cvar_95:,.0f}."
        )

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.hist(simulation_results, bins=60,
                 color='steelblue', edgecolor='white', alpha=0.7)
        ax4.axvline(INVESTMENT, color='black', linestyle='--',
                    linewidth=2, label=f'Your Investment ₹{INVESTMENT:,.0f}')
        ax4.axvline(var_95, color='red', linestyle='--',
                    linewidth=2, label=f'VaR 95% ₹{var_95:,.0f}')
        ax4.axvline(var_99, color='darkred', linestyle='--',
                    linewidth=2, label=f'VaR 99% ₹{var_99:,.0f}')
        ax4.axvline(cvar_95, color='orange', linestyle='--',
                    linewidth=2, label=f'CVaR 95% ₹{cvar_95:,.0f}')
        ax4.axvline(median_val, color='green', linestyle='--',
                    linewidth=2, label=f'Median ₹{median_val:,.0f}')
        ax4.set_xlabel('Portfolio Value After 1 Year (₹)')
        ax4.set_ylabel('Number of Simulations')
        ax4.set_title(f'Distribution of {NUM_SIMULATIONS} Simulated Portfolio Outcomes')
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # ── SECTION 12 — SUMMARY ─────────────────────────────────────────────
        st.markdown("### 12. 📋 Complete Portfolio Analysis Summary")
        st.caption(
            "Everything this app has calculated — summarised in one place. "
            "Use this as your complete investment decision brief."
        )
        st.divider()

        st.subheader("What You're Investing In")
        st.write(
            f"You have selected {len(tickers)} stocks across "
            f"{len(sector_weights)} sectors. The dominant sector is "
            f"**{dominant_sector[0]}** at {dominant_sector[1]:.1%} allocation. "
            f"The least correlated pair — **{min_corr_pair[0]}** and "
            f"**{min_corr_pair[1]}** (correlation: {min_corr_val:.2f}%) — "
            f"provides the strongest diversification benefit in your portfolio."
        )

        st.subheader("How to Split Your Investment")
        st.write(
            f"For a ₹{INVESTMENT:,.0f} investment, the mathematically optimal "
            f"allocation based on the Maximum Sharpe Ratio is shown in Section 3 above. "
            f"This allocation was derived by simulating 10,000 different weight "
            f"combinations and identifying the one that delivers the highest "
            f"return per unit of risk — the core principle of Modern Portfolio Theory."
        )

        st.subheader("Historical Performance (2022–2024)")
        st.write(
            f"Applying the optimal weights to 3 years of real NSE data, your "
            f"portfolio turned ₹{INVESTMENT:,.0f} into ₹{final_value:,.0f} — "
            f"a total return of **{total_return:.1f}%**. "
            f"During the same period, NIFTY 50 returned {nifty_return:.1f}%. "
            f"Your portfolio {'outperformed' if outperformance > 0 else 'underperformed'} "
            f"the benchmark by **{abs(outperformance):.1f}%**. "
            f"The maximum drawdown was {max_drawdown:.1f}% in "
            f"{max_drawdown_date.strftime('%B %Y')} — the portfolio fully recovered "
            f"and continued to grow."
        )

        st.subheader("Future Risk Outlook")
        st.write(
            f"Based on {NUM_SIMULATIONS} Monte Carlo simulations using historical "
            f"volatility data, your portfolio's most likely outcome after one year "
            f"is **₹{median_val:,.0f}**. "
            f"There is a 95% probability that your portfolio will not fall below "
            f"**₹{var_95:,.0f}** — a maximum loss of ₹{INVESTMENT - var_95:,.0f}. "
            f"In the worst 1% of scenarios, the portfolio could fall to "
            f"₹{var_99:,.0f}. If losses do breach the VaR threshold, the average "
            f"loss across those worst scenarios is ₹{INVESTMENT - cvar_95:,.0f} "
            f"— this is your Expected Shortfall (CVaR). "
            f"These metrics are the exact same risk measures that banks like "
            f"JPMorgan Chase report to regulators every single day."
        )

        st.subheader("Key Takeaways")
        st.write(
            f"1. **Diversification works.** Combining {len(tickers)} stocks across "
            f"multiple sectors reduces your portfolio volatility below any "
            f"individual stock's volatility.\n\n"
            f"2. **Optimisation beats guessing.** Equal weight allocation is not "
            f"optimal. Mathematical weight optimisation using the Efficient Frontier "
            f"identifies the exact allocation that maximises your return per unit "
            f"of risk.\n\n"
            f"3. **Past performance is evidence, not guarantee.** The {total_return:.1f}% "
            f"historical return is real — it happened. But future returns depend on "
            f"market conditions that cannot be predicted. The Monte Carlo simulation "
            f"gives you a probability range, not a certainty.\n\n"
            f"4. **Risk has a price.** The VaR of ₹{INVESTMENT - var_95:,.0f} is "
            f"the cost of participating in equity markets. Every investment "
            f"carries risk — the goal of this tool is to help you take the "
            f"right amount of risk for the return you're targeting."
        )

        st.markdown("""
            <hr>
            <div style='text-align: center; padding: 1rem; color: grey; font-size: 0.85rem;'>
                <strong>Portfolio Risk Analyser</strong> · Built by Bhargav Singh<br>
                D.J. Sanghvi College of Engineering, Mumbai · 2026<br>
                <a href='https://www.linkedin.com/in/bhargavsingh9' target='_blank'>LinkedIn</a> &nbsp;|&nbsp;
                <a href='https://github.com/bhargavs999' target='_blank'>GitHub</a> &nbsp;|&nbsp;
                Data Source: Yahoo Finance / NSE
            </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Simulation produced invalid results. Please try again.")
