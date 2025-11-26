#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize

plt.style.use("seaborn-v0_8")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "XOM", "JNJ", "PG"
]

START_DATE = "2015-01-01"
END_DATE = None

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.02


# In[2]:


def download_prices(tickers, start, end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data["Close"]
    return prices.dropna()

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def annualize_mean_cov(daily_returns):
    mu_daily = daily_returns.mean().values
    cov_daily = daily_returns.cov().values
    mu_annual = mu_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS
    return mu_annual, cov_annual

def portfolio_performance(weights, mu, cov, rf=0.0):
    w = np.array(weights)
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


# In[3]:


def equal_weight(n):
    return np.ones(n) / n

def min_variance_portfolio(cov):
    n = cov.shape[0]
    def var(w, cov):
        return w @ cov @ w
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(var, x0=x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# In[4]:


def black_litterman(mu, cov, market_weights, tau=0.05, delta=2.5, P=None, Q=None, omega=None):
    mu = np.array(mu)
    cov = np.array(cov)
    w_mkt = np.array(market_weights)
    pi = delta * cov @ w_mkt
    if P is None or Q is None:
        return pi
    P = np.array(P)
    Q = np.array(Q)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
    tau_cov_inv = np.linalg.inv(tau * cov)
    omega_inv = np.linalg.inv(omega)
    middle = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    right = tau_cov_inv @ pi + P.T @ omega_inv @ Q
    return middle @ right


# In[5]:


def equity_curve(weights, daily_returns):
    w = np.array(weights)
    port_rets = daily_returns.values @ w
    eq = (1 + port_rets).cumprod()
    return pd.Series(eq, index=daily_returns.index)

def plot_equity_curves(curves_dict, title=""):
    plt.figure(figsize=(10, 6))
    for name, eq in curves_dict.items():
        plt.plot(eq.index, eq.values, label=name)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=300)
    plt.show()


# In[6]:


prices = download_prices(TICKERS, START_DATE, END_DATE)
display(prices.tail())

daily_rets = compute_log_returns(prices)
mu, cov = annualize_mean_cov(daily_rets)

n = len(TICKERS)
rf = RISK_FREE_ANNUAL

w_eq = equal_weight(n)
w_mkv = min_variance_portfolio(cov)


# In[7]:


w_mkt = w_eq.copy()

P = []
Q = []

p1 = np.zeros(n)
p1[TICKERS.index("AAPL")] = 1.0
p1 -= w_mkt
P.append(p1)
Q.append(0.03)

P = np.array(P)
Q = np.array(Q)

mu_bl = black_litterman(mu, cov, w_mkt, P=P, Q=Q)


# In[8]:


ret_eq,  vol_eq,  sh_eq  = portfolio_performance(w_eq,  mu,    cov, rf)
ret_mkv, vol_mkv, sh_mkv = portfolio_performance(w_mkv, mu,    cov, rf)
ret_bl,  vol_bl,  sh_bl  = portfolio_performance(w_mkv, mu_bl, cov, rf)

weights_df = pd.DataFrame({
    "Ticker": TICKERS,
    "Equal": w_eq,
    "Markowitz": w_mkv
}).set_index("Ticker")

stats_df = pd.DataFrame({
    "Return":     [ret_eq,  ret_mkv,  ret_bl],
    "Volatility": [vol_eq,  vol_mkv,  vol_bl],
    "Sharpe":     [sh_eq,   sh_mkv,   sh_bl],
}, index=["Equal", "Markowitz", "Markowitz + BL(mu)"])

display(weights_df.round(4))
display(stats_df.round(4))


# In[9]:


eq_eq  = equity_curve(w_eq,  daily_rets)
eq_mkv = equity_curve(w_mkv, daily_rets)

plot_equity_curves(
    {
        "Equal": eq_eq,
        "Markowitz_minVar": eq_mkv,
    },
    title="Portfolio comparison"
)

