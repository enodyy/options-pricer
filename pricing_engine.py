"""
美股期权定价核心引擎
支持: Black-Scholes 欧式期权定价、二叉树美式期权定价、Greeks 计算、隐含波动率求解
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionResult:
    """期权定价结果"""
    price: float
    delta: float
    gamma: float
    theta: float  # 每日
    vega: float   # 每1%波动率变化
    rho: float    # 每1%利率变化
    intrinsic: float
    time_value: float


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    """计算 Black-Scholes d1, d2"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(
    S: float,       # 标的价格
    K: float,       # 行权价
    T: float,       # 到期时间（年）
    r: float,       # 无风险利率（年化，如0.05）
    q: float,       # 股息率（年化，如0.02）
    sigma: float,   # 波动率（年化，如0.25）
    option_type: Literal["call", "put"] = "call"
) -> OptionResult:
    """
    Black-Scholes 欧式期权定价 + Greeks
    """
    if T <= 0:
        # 到期时
        if option_type == "call":
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        return OptionResult(
            price=intrinsic, delta=1.0 if intrinsic > 0 else 0.0,
            gamma=0, theta=0, vega=0, rho=0,
            intrinsic=intrinsic, time_value=0
        )

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    if option_type == "call":
        price = S * exp_qT * norm.cdf(d1) - K * exp_rT * norm.cdf(d2)
        delta = exp_qT * norm.cdf(d1)
        intrinsic = max(S - K, 0)
        rho_val = K * T * exp_rT * norm.cdf(d2) / 100
    else:
        price = K * exp_rT * norm.cdf(-d2) - S * exp_qT * norm.cdf(-d1)
        delta = -exp_qT * norm.cdf(-d1)
        intrinsic = max(K - S, 0)
        rho_val = -K * T * exp_rT * norm.cdf(-d2) / 100

    gamma = exp_qT * norm.pdf(d1) / (S * sigma * sqrt_T)
    theta = (-(S * sigma * exp_qT * norm.pdf(d1)) / (2 * sqrt_T)
             - r * K * exp_rT * norm.cdf(d2 if option_type == "call" else -d2)
                * (1 if option_type == "call" else -1)
             + q * S * exp_qT * norm.cdf(d1 if option_type == "call" else -d1)
                * (1 if option_type == "call" else -1)
             ) / 365  # 每日theta
    vega = S * exp_qT * norm.pdf(d1) * sqrt_T / 100  # 每1%波动率

    return OptionResult(
        price=price, delta=delta, gamma=gamma, theta=theta,
        vega=vega, rho=rho_val,
        intrinsic=intrinsic, time_value=price - intrinsic
    )


def binomial_american(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: Literal["call", "put"] = "put",
    steps: int = 200
) -> float:
    """
    CRR 二叉树模型，用于美式期权定价
    美式 Put 通常比欧式贵（提前行权价值），美式 Call 无分红时 = 欧式 Call
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # 终端节点的标的价格
    ST = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1, 1)

    if option_type == "call":
        values = np.maximum(ST - K, 0)
    else:
        values = np.maximum(K - ST, 0)

    # 倒推
    for i in range(steps - 1, -1, -1):
        ST_i = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1, 1)
        hold = disc * (p * values[:i + 1] + (1 - p) * values[1:i + 2])
        if option_type == "call":
            exercise = np.maximum(ST_i - K, 0)
        else:
            exercise = np.maximum(K - ST_i, 0)
        values = np.maximum(hold, exercise)

    return float(values[0])


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: Literal["call", "put"] = "call"
) -> float:
    """
    通过 Brent 方法反解隐含波动率
    """
    if T <= 0:
        return 0.0

    def objective(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type).price - market_price

    try:
        iv = brentq(objective, 0.001, 5.0, xtol=1e-8)
        return iv
    except ValueError:
        return np.nan


def compute_iv_surface(
    S: float,
    r: float,
    q: float,
    strikes: list[float],
    expirations: list[float],
    market_prices: dict[tuple[float, float], float],
    option_type: Literal["call", "put"] = "call"
) -> np.ndarray:
    """
    计算隐含波动率曲面
    market_prices: {(strike, expiry): price}
    返回: 2D array [len(strikes) x len(expirations)]
    """
    surface = np.full((len(strikes), len(expirations)), np.nan)
    for i, K in enumerate(strikes):
        for j, T in enumerate(expirations):
            key = (K, T)
            if key in market_prices:
                iv = implied_volatility(market_prices[key], S, K, T, r, q, option_type)
                surface[i, j] = iv
    return surface


def option_pnl_at_expiry(
    S_range: np.ndarray,
    positions: list[dict]
) -> np.ndarray:
    """
    计算期权组合到期日的盈亏曲线
    positions: [{"type": "call"/"put", "strike": K, "premium": p, "qty": n, "side": "buy"/"sell"}, ...]
    """
    pnl = np.zeros_like(S_range, dtype=float)
    for pos in positions:
        K = pos["strike"]
        premium = pos["premium"]
        qty = pos["qty"]
        multiplier = 1 if pos["side"] == "buy" else -1

        if pos["type"] == "call":
            payoff = np.maximum(S_range - K, 0)
        else:
            payoff = np.maximum(K - S_range, 0)

        pnl += multiplier * qty * (payoff - premium) * 100  # 每手100股

    return pnl


# ========== 常用策略组合 ==========

def bull_call_spread(S, K1, K2, T, r, q, sigma):
    """牛市看涨价差：买低行权价Call + 卖高行权价Call"""
    long = black_scholes(S, K1, T, r, q, sigma, "call")
    short = black_scholes(S, K2, T, r, q, sigma, "call")
    net_premium = long.price - short.price
    return {
        "net_premium": net_premium,
        "max_profit": (K2 - K1) - net_premium,
        "max_loss": net_premium,
        "breakeven": K1 + net_premium,
        "long_leg": long,
        "short_leg": short,
    }


def bear_put_spread(S, K1, K2, T, r, q, sigma):
    """熊市看跌价差：买高行权价Put + 卖低行权价Put"""
    long = black_scholes(S, K2, T, r, q, sigma, "put")
    short = black_scholes(S, K1, T, r, q, sigma, "put")
    net_premium = long.price - short.price
    return {
        "net_premium": net_premium,
        "max_profit": (K2 - K1) - net_premium,
        "max_loss": net_premium,
        "breakeven": K2 - net_premium,
        "long_leg": long,
        "short_leg": short,
    }


def long_straddle(S, K, T, r, q, sigma):
    """买入跨式：同时买入同行权价的Call和Put"""
    call = black_scholes(S, K, T, r, q, sigma, "call")
    put = black_scholes(S, K, T, r, q, sigma, "put")
    total_premium = call.price + put.price
    return {
        "total_premium": total_premium,
        "breakeven_up": K + total_premium,
        "breakeven_down": K - total_premium,
        "call": call,
        "put": put,
    }


def iron_condor(S, K1, K2, K3, K4, T, r, q, sigma):
    """铁鹰策略: 卖K2 Put + 买K1 Put + 卖K3 Call + 买K4 Call (K1<K2<K3<K4)"""
    buy_put = black_scholes(S, K1, T, r, q, sigma, "put")
    sell_put = black_scholes(S, K2, T, r, q, sigma, "put")
    sell_call = black_scholes(S, K3, T, r, q, sigma, "call")
    buy_call = black_scholes(S, K4, T, r, q, sigma, "call")

    net_credit = (sell_put.price - buy_put.price) + (sell_call.price - buy_call.price)
    max_loss = max(K2 - K1, K4 - K3) - net_credit
    return {
        "net_credit": net_credit,
        "max_profit": net_credit,
        "max_loss": max_loss,
        "breakeven_down": K2 - net_credit,
        "breakeven_up": K3 + net_credit,
    }
