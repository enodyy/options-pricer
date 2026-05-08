"""
市场数据获取模块
- T-Bill 无风险利率（自动匹配期限）
- 股票收盘价
- TTM 股息率
- 历史波动率（多周期）

使用 yf.download() 代替 Ticker.history()，更稳定，不容易触发限流。
"""

from __future__ import annotations

import yfinance as yf
import numpy as np
import datetime
import time


def _download_history(symbol: str, period: str = "1y") -> "pd.DataFrame | None":
    """
    用 yf.download 获取历史数据，带重试机制。
    比 Ticker.history() 更不容易被限流。
    """
    for attempt in range(3):
        try:
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1)
    return None


def fetch_tbill_rate(days_to_expiry: int = 30) -> float | None:
    """
    根据期权到期天数，获取最匹配期限的T-Bill利率
    返回: 年化利率（如 0.043 表示 4.3%），失败返回 None
    """
    ticker_symbol = "^IRX" if days_to_expiry <= 180 else "^FVX"

    try:
        df = _download_history(ticker_symbol, period="5d")
        if df is None or df.empty:
            return None
        close_val = df["Close"].iloc[-1]
        rate = float(close_val.iloc[0]) if hasattr(close_val, 'iloc') else float(close_val)
        return rate / 100
    except Exception:
        return None


def fetch_stock_data(symbol: str) -> dict:
    """
    获取股票的收盘价、TTM股息率、历史波动率
    """
    result = {
        "price": None,
        "dividend_yield": 0.0,
        "name": symbol,
        "error": None,
        "hv": {},
    }

    try:
        # 用 yf.download 获取1年历史数据（更稳定）
        hist = _download_history(symbol, period="1y")
        if hist is None or hist.empty:
            result["error"] = f"无法获取 {symbol} 的行情数据，请检查股票代码或稍后重试"
            return result

        # 获取收盘价（处理单层/多层列名）
        close_col = hist["Close"]
        if hasattr(close_col, 'columns'):
            # 多层列名 (MultiIndex)，取第一列
            closes = close_col.iloc[:, 0].values
        else:
            closes = close_col.values
        result["price"] = float(closes[-1])

        # 计算多周期历史波动率
        if len(closes) > 20:
            log_returns = np.log(closes[1:] / closes[:-1])
            # 去掉 NaN
            log_returns = log_returns[~np.isnan(log_returns)]
            for window, label in [(20, "20日"), (60, "60日"), (120, "120日"), (252, "年")]:
                if len(log_returns) >= window:
                    recent = log_returns[-window:]
                    hv = float(np.std(recent, ddof=1) * np.sqrt(252))
                    result["hv"][label] = hv

        # 获取TTM股息（单独用Ticker，因为download不含股息明细）
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            if len(dividends) > 0:
                one_year_ago = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=365)
                recent_divs = dividends[dividends.index >= one_year_ago]
                ttm_dividend = float(recent_divs.sum()) if len(recent_divs) > 0 else 0.0
                if ttm_dividend > 0 and result["price"] > 0:
                    result["dividend_yield"] = ttm_dividend / result["price"]
        except Exception:
            result["dividend_yield"] = 0.0

        # 获取公司名称
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            result["name"] = info.get("shortName", symbol)
        except Exception:
            result["name"] = symbol

    except Exception as e:
        result["error"] = f"获取数据失败: {str(e)}"

    return result
