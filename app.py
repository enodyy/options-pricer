"""
美股期权定价分析工具 - Streamlit Web 界面
功能：期权定价、Greeks分析、隐含波动率、策略组合分析、盈亏可视化
自动获取：T-Bill利率、股票收盘价、TTM股息率
"""

import streamlit as st
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pricing_engine import (
    black_scholes, binomial_american, implied_volatility,
    option_pnl_at_expiry, bull_call_spread, bear_put_spread,
    long_straddle, iron_condor
)
from data_fetcher import fetch_stock_data, fetch_tbill_rate

st.set_page_config(page_title="期权定价分析", page_icon="📊", layout="wide")

st.title("📊 美股期权定价分析工具")

# ========== 侧边栏：公共参数 ==========
st.sidebar.header("📌 标的股票")
ticker_input = st.sidebar.text_input("股票代码", value="AAPL", help="输入美股代码，如 AAPL、TSLA、MSFT")

# 获取数据按钮
if st.sidebar.button("🔄 获取市场数据", type="primary") or "stock_data" not in st.session_state:
    with st.sidebar:
        with st.spinner(f"正在获取 {ticker_input} 数据..."):
            stock_data = fetch_stock_data(ticker_input.upper().strip())
            st.session_state["stock_data"] = stock_data
            st.session_state["ticker"] = ticker_input.upper().strip()

        with st.spinner("正在获取 T-Bill 利率..."):
            tbill = fetch_tbill_rate(30)
            st.session_state["tbill_rate"] = tbill

# 从 session_state 读取，设定默认值
stock_data = st.session_state.get("stock_data", {})
default_price = stock_data.get("price") or 150.0
default_div_yield = (stock_data.get("dividend_yield") or 0.0) * 100  # 转为百分比
default_tbill = (st.session_state.get("tbill_rate") or 0.045) * 100  # 转为百分比
stock_name = stock_data.get("name", ticker_input)

# 波动率数据
hv_data = stock_data.get("hv", {})
default_hv = round(hv_data.get("20日", 0.25) * 100, 1)  # 默认用20日HV

# 显示自动获取的信息
if stock_data.get("price"):
    hv_display = " / ".join([f"{k}: {v*100:.1f}%" for k, v in hv_data.items()])
    st.sidebar.success(
        f"**{stock_name}**  \n"
        f"收盘价: ${default_price:.2f}  \n"
        f"TTM股息率: {default_div_yield:.2f}%  \n"
        f"历史波动率: {hv_display or '无数据'}"
    )
elif stock_data.get("error"):
    st.sidebar.error(stock_data["error"])

if st.session_state.get("tbill_rate"):
    st.sidebar.info(f"13周 T-Bill: {default_tbill:.2f}%")
else:
    st.sidebar.warning("T-Bill 利率获取失败，使用默认值 4.5%")

st.sidebar.markdown("---")
st.sidebar.header("📌 参数（可手动调整）")
S = st.sidebar.number_input("标的股价 ($)", value=round(default_price, 2), min_value=0.01, step=1.0)
r = st.sidebar.number_input("无风险利率 (%)", value=round(default_tbill, 2), step=0.1,
                              help="自动获取13周T-Bill利率，短期期权最匹配") / 100
q = st.sidebar.number_input("股息率 - TTM (%)", value=round(default_div_yield, 2), step=0.1,
                              help="自动获取TTM股息率") / 100

# 波动率选择
st.sidebar.markdown("---")
st.sidebar.header("📌 波动率")
if hv_data:
    hv_options = {f"{k} HV: {v*100:.1f}%": round(v * 100, 1) for k, v in hv_data.items()}
    hv_options["手动输入"] = None
    hv_choice = st.sidebar.radio("选择波动率来源", list(hv_options.keys()), index=0,
                                  help="20日HV适合短期期权，60日HV适合中期期权。也可手动输入IV")
    if hv_options[hv_choice] is not None:
        default_sigma = hv_options[hv_choice]
    else:
        default_sigma = default_hv
else:
    default_sigma = 25.0
    st.sidebar.caption("未获取到历史数据，请手动输入波动率")


# ========== 通用的到期日输入组件 ==========
def expiry_date_input(label_prefix: str, key_suffix: str, default_days: int = 30):
    """统一的到期日输入：选择日期，自动计算天数"""
    today = datetime.date.today()
    default_date = today + datetime.timedelta(days=default_days)

    expiry_date = st.date_input(
        f"{label_prefix}到期日",
        value=default_date,
        min_value=today + datetime.timedelta(days=1),
        key=f"expiry_{key_suffix}",
        help="选择期权到期日，自动计算剩余天数"
    )

    days = (expiry_date - today).days
    st.caption(f"距到期 **{days}** 天 ({days/365:.3f} 年)")
    return days, days / 365


# ========== 标签页 ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔢 单个期权定价", "📈 Greeks 分析", "🌊 隐含波动率",
    "🎯 策略组合", "📚 期权入门"
])

# =============================================
# Tab 1: 单个期权定价
# =============================================
with tab1:
    st.subheader("Black-Scholes / 二叉树 期权定价")

    col1, col2 = st.columns(2)
    with col1:
        K = st.number_input("行权价 ($)", value=float(round(S)), min_value=0.01, step=1.0, key="t1_K")
        days_to_expiry, T = expiry_date_input("", "t1")
    with col2:
        sigma = st.number_input("波动率 (%)", value=default_sigma, min_value=0.1, step=1.0, key="t1_sigma") / 100
        option_type = st.selectbox("期权类型", ["call (看涨)", "put (看跌)"], key="t1_type")
        opt_type = "call" if "call" in option_type else "put"

    # 计算
    bs_result = black_scholes(S, K, T, r, q, sigma, opt_type)
    american_price = binomial_american(S, K, T, r, q, sigma, opt_type)

    st.markdown("---")

    # 显示结果
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 💰 期权价格")
        st.metric("BS 理论价 (欧式)", f"${bs_result.price:.4f}")
        st.metric("美式期权价 (二叉树)", f"${american_price:.4f}")
        early_exercise = american_price - bs_result.price
        if early_exercise > 0.001:
            st.info(f"提前行权价值: ${early_exercise:.4f}")
        st.metric("内在价值", f"${bs_result.intrinsic:.4f}")
        st.metric("时间价值", f"${bs_result.time_value:.4f}")

    with col_b:
        st.markdown("### 📐 Greeks")
        st.metric("Delta (Δ)", f"{bs_result.delta:.4f}",
                   help="标的价格变动$1时，期权价格变动量")
        st.metric("Gamma (Γ)", f"{bs_result.gamma:.4f}",
                   help="标的价格变动$1时，Delta变动量")
        st.metric("Theta (Θ)", f"{bs_result.theta:.4f}",
                   help="每过1天，期权价值衰减量")

    with col_c:
        st.markdown("### 📐 Greeks (续)")
        st.metric("Vega (ν)", f"{bs_result.vega:.4f}",
                   help="波动率上升1%时，期权价格变动量")
        st.metric("Rho (ρ)", f"{bs_result.rho:.4f}",
                   help="利率上升1%时，期权价格变动量")

    # 对照表格：不同行权价
    st.markdown("---")
    st.subheader("不同行权价对比")
    strike_range = np.arange(max(S * 0.85, 1), S * 1.15, S * 0.025)
    table_data = []
    for k in strike_range:
        res = black_scholes(S, k, T, r, q, sigma, opt_type)
        moneyness = "ITM" if (opt_type == "call" and S > k) or (opt_type == "put" and S < k) else \
                    "ATM" if abs(S - k) < S * 0.01 else "OTM"
        table_data.append({
            "行权价": f"${k:.1f}",
            "状态": moneyness,
            "期权价": f"${res.price:.3f}",
            "Delta": f"{res.delta:.3f}",
            "Gamma": f"{res.gamma:.4f}",
            "Theta": f"{res.theta:.4f}",
            "Vega": f"{res.vega:.4f}",
            "内在价值": f"${res.intrinsic:.2f}",
            "时间价值": f"${res.time_value:.3f}",
        })
    st.dataframe(table_data, use_container_width=True)


# =============================================
# Tab 2: Greeks 可视化
# =============================================
with tab2:
    st.subheader("Greeks 随标的价格变化")

    col1, col2 = st.columns(2)
    with col1:
        K2 = st.number_input("行权价 ($)", value=float(round(S)), min_value=0.01, step=1.0, key="t2_K")
        days2, T2 = expiry_date_input("", "t2")
    with col2:
        sigma2 = st.number_input("波动率 (%)", value=default_sigma, min_value=0.1, step=1.0, key="t2_sigma") / 100
        opt_type2 = "call" if "call" in st.selectbox("期权类型", ["call (看涨)", "put (看跌)"], key="t2_type") else "put"

    S_range = np.linspace(S * 0.7, S * 1.3, 200)

    deltas, gammas, thetas, vegas, prices = [], [], [], [], []
    for s in S_range:
        res = black_scholes(s, K2, T2, r, q, sigma2, opt_type2)
        deltas.append(res.delta)
        gammas.append(res.gamma)
        thetas.append(res.theta)
        vegas.append(res.vega)
        prices.append(res.price)

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("期权价格", "Delta (Δ)", "Gamma (Γ)",
                                       "Theta (Θ) 每日", "Vega (ν) 每1%", "盈亏图"))

    fig.add_trace(go.Scatter(x=S_range, y=prices, name="价格", line=dict(color="#2196F3")), row=1, col=1)
    fig.add_vline(x=K2, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=S_range, y=deltas, name="Delta", line=dict(color="#4CAF50")), row=1, col=2)
    fig.add_trace(go.Scatter(x=S_range, y=gammas, name="Gamma", line=dict(color="#FF9800")), row=2, col=1)
    fig.add_trace(go.Scatter(x=S_range, y=thetas, name="Theta", line=dict(color="#F44336")), row=2, col=2)
    fig.add_trace(go.Scatter(x=S_range, y=vegas, name="Vega", line=dict(color="#9C27B0")), row=3, col=1)

    # 到期盈亏
    buy_premium = black_scholes(S, K2, T2, r, q, sigma2, opt_type2).price
    if opt_type2 == "call":
        pnl_expiry = np.maximum(S_range - K2, 0) - buy_premium
    else:
        pnl_expiry = np.maximum(K2 - S_range, 0) - buy_premium
    pnl_expiry *= 100  # 每手
    fig.add_trace(go.Scatter(x=S_range, y=pnl_expiry, name="到期盈亏/手",
                             line=dict(color="#607D8B")), row=3, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)

    fig.update_layout(height=900, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 不同到期日对比
    st.subheader("Theta 衰减：不同到期日的期权价格")
    days_list = [7, 14, 30, 60, 90]
    fig_decay = go.Figure()
    for d in days_list:
        t = d / 365
        ps = [black_scholes(s, K2, t, r, q, sigma2, opt_type2).price for s in S_range]
        fig_decay.add_trace(go.Scatter(x=S_range, y=ps, name=f"{d}天"))
    # 到期内在价值
    if opt_type2 == "call":
        intrinsic_line = np.maximum(S_range - K2, 0)
    else:
        intrinsic_line = np.maximum(K2 - S_range, 0)
    fig_decay.add_trace(go.Scatter(x=S_range, y=intrinsic_line, name="到期 (内在价值)",
                                    line=dict(dash="dash", color="black")))
    fig_decay.update_layout(
        height=400, template="plotly_white",
        xaxis_title="标的价格 ($)", yaxis_title="期权价格 ($)",
        title="时间价值衰减可视化"
    )
    st.plotly_chart(fig_decay, use_container_width=True)


# =============================================
# Tab 3: 隐含波动率
# =============================================
with tab3:
    st.subheader("隐含波动率计算 & 波动率微笑")

    st.markdown("#### 单个期权隐含波动率")
    col1, col2 = st.columns(2)
    with col1:
        K3 = st.number_input("行权价 ($)", value=float(round(S)), min_value=0.01, step=1.0, key="t3_K")
        days3, T3 = expiry_date_input("", "t3")
    with col2:
        market_price = st.number_input("市场价格 ($)", value=5.0, min_value=0.01, step=0.1, key="t3_mp")
        opt_type3 = "call" if "call" in st.selectbox("期权类型", ["call (看涨)", "put (看跌)"], key="t3_type") else "put"

    iv = implied_volatility(market_price, S, K3, T3, r, q, opt_type3)

    if not np.isnan(iv):
        st.success(f"**隐含波动率: {iv*100:.2f}%**")
        verify = black_scholes(S, K3, T3, r, q, iv, opt_type3)
        st.caption(f"验证: IV={iv*100:.2f}% → 理论价=${verify.price:.4f} vs 市场价=${market_price:.4f}")
    else:
        st.error("无法求解隐含波动率，请检查输入参数（市场价格可能超出合理范围）")

    st.markdown("---")

    # 波动率微笑模拟
    st.subheader("波动率微笑 / 偏斜 (Skew) 可视化")
    st.markdown("输入不同行权价的市场价格，观察隐含波动率曲线形态")

    num_strikes = st.slider("行权价数量", 5, 15, 9)
    strike_center = S
    strike_step = S * 0.025

    strikes_smile = [strike_center + (i - num_strikes // 2) * strike_step for i in range(num_strikes)]

    col_inputs = st.columns(min(num_strikes, 5))
    smile_prices = {}

    for i, k in enumerate(strikes_smile):
        with col_inputs[i % len(col_inputs)]:
            dist = abs(k - S) / S
            skew_sigma = sigma2 * (1 + 0.5 * dist)
            default_price_smile = black_scholes(S, k, T3, r, q, skew_sigma, opt_type3).price
            p = st.number_input(f"K=${k:.0f}", value=round(default_price_smile, 2),
                                min_value=0.01, step=0.1, key=f"smile_{i}")
            smile_prices[k] = p

    ivs = []
    valid_strikes = []
    for k in strikes_smile:
        iv_val = implied_volatility(smile_prices[k], S, k, T3, r, q, opt_type3)
        if not np.isnan(iv_val):
            ivs.append(iv_val * 100)
            valid_strikes.append(k)

    if valid_strikes:
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=valid_strikes, y=ivs, mode="lines+markers",
            name="隐含波动率", line=dict(color="#2196F3", width=3),
            marker=dict(size=8)
        ))
        fig_smile.add_vline(x=S, line_dash="dash", line_color="red",
                            annotation_text=f"现价 ${S:.0f}")
        fig_smile.update_layout(
            height=400, template="plotly_white",
            xaxis_title="行权价 ($)", yaxis_title="隐含波动率 (%)",
            title="波动率微笑 (Volatility Smile)"
        )
        st.plotly_chart(fig_smile, use_container_width=True)


# =============================================
# Tab 4: 策略组合
# =============================================
with tab4:
    st.subheader("常用期权策略组合分析")

    strategy = st.selectbox("选择策略", [
        "牛市看涨价差 (Bull Call Spread)",
        "熊市看跌价差 (Bear Put Spread)",
        "买入跨式 (Long Straddle)",
        "铁鹰策略 (Iron Condor)",
        "自定义组合"
    ])

    sigma4 = st.number_input("波动率 (%)", value=default_sigma, min_value=0.1, step=1.0, key="t4_sigma") / 100
    days4, T4 = expiry_date_input("", "t4")

    if strategy == "牛市看涨价差 (Bull Call Spread)":
        col1, col2 = st.columns(2)
        with col1:
            K_low = st.number_input("低行权价 (买入Call)", value=round(S * 0.97, 1), step=1.0, key="bull_k1")
        with col2:
            K_high = st.number_input("高行权价 (卖出Call)", value=round(S * 1.03, 1), step=1.0, key="bull_k2")

        result = bull_call_spread(S, K_low, K_high, T4, r, q, sigma4)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("净权利金成本", f"${result['net_premium']:.2f}")
        col_b.metric("最大盈利", f"${result['max_profit']:.2f}")
        col_c.metric("最大亏损", f"${result['max_loss']:.2f}")
        col_d.metric("盈亏平衡", f"${result['breakeven']:.2f}")

        positions = [
            {"type": "call", "strike": K_low, "premium": result['long_leg'].price, "qty": 1, "side": "buy"},
            {"type": "call", "strike": K_high, "premium": result['short_leg'].price, "qty": 1, "side": "sell"},
        ]

    elif strategy == "熊市看跌价差 (Bear Put Spread)":
        col1, col2 = st.columns(2)
        with col1:
            K_low = st.number_input("低行权价 (卖出Put)", value=round(S * 0.97, 1), step=1.0, key="bear_k1")
        with col2:
            K_high = st.number_input("高行权价 (买入Put)", value=round(S * 1.03, 1), step=1.0, key="bear_k2")

        result = bear_put_spread(S, K_low, K_high, T4, r, q, sigma4)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("净权利金成本", f"${result['net_premium']:.2f}")
        col_b.metric("最大盈利", f"${result['max_profit']:.2f}")
        col_c.metric("最大亏损", f"${result['max_loss']:.2f}")
        col_d.metric("盈亏平衡", f"${result['breakeven']:.2f}")

        positions = [
            {"type": "put", "strike": K_high, "premium": result['long_leg'].price, "qty": 1, "side": "buy"},
            {"type": "put", "strike": K_low, "premium": result['short_leg'].price, "qty": 1, "side": "sell"},
        ]

    elif strategy == "买入跨式 (Long Straddle)":
        K_straddle = st.number_input("行权价", value=float(round(S)), step=1.0, key="straddle_k")
        result = long_straddle(S, K_straddle, T4, r, q, sigma4)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("总权利金成本", f"${result['total_premium']:.2f}")
        col_b.metric("上方盈亏平衡", f"${result['breakeven_up']:.2f}")
        col_c.metric("下方盈亏平衡", f"${result['breakeven_down']:.2f}")

        positions = [
            {"type": "call", "strike": K_straddle, "premium": result['call'].price, "qty": 1, "side": "buy"},
            {"type": "put", "strike": K_straddle, "premium": result['put'].price, "qty": 1, "side": "buy"},
        ]

    elif strategy == "铁鹰策略 (Iron Condor)":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            K_ic1 = st.number_input("K1 (买Put)", value=round(S * 0.92, 1), step=1.0, key="ic_k1")
        with col2:
            K_ic2 = st.number_input("K2 (卖Put)", value=round(S * 0.96, 1), step=1.0, key="ic_k2")
        with col3:
            K_ic3 = st.number_input("K3 (卖Call)", value=round(S * 1.04, 1), step=1.0, key="ic_k3")
        with col4:
            K_ic4 = st.number_input("K4 (买Call)", value=round(S * 1.08, 1), step=1.0, key="ic_k4")

        result = iron_condor(S, K_ic1, K_ic2, K_ic3, K_ic4, T4, r, q, sigma4)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("净收入权利金", f"${result['net_credit']:.2f}")
        col_b.metric("最大盈利", f"${result['max_profit']:.2f}")
        col_c.metric("最大亏损", f"${result['max_loss']:.2f}")
        col_d.metric("盈亏区间", f"${result['breakeven_down']:.1f} - ${result['breakeven_up']:.1f}")

        bp = black_scholes(S, K_ic1, T4, r, q, sigma4, "put")
        sp = black_scholes(S, K_ic2, T4, r, q, sigma4, "put")
        sc = black_scholes(S, K_ic3, T4, r, q, sigma4, "call")
        bc = black_scholes(S, K_ic4, T4, r, q, sigma4, "call")
        positions = [
            {"type": "put", "strike": K_ic1, "premium": bp.price, "qty": 1, "side": "buy"},
            {"type": "put", "strike": K_ic2, "premium": sp.price, "qty": 1, "side": "sell"},
            {"type": "call", "strike": K_ic3, "premium": sc.price, "qty": 1, "side": "sell"},
            {"type": "call", "strike": K_ic4, "premium": bc.price, "qty": 1, "side": "buy"},
        ]

    else:  # 自定义组合
        st.markdown("自由添加期权腿，构建任意策略组合")
        num_legs = st.number_input("期权腿数量", value=2, min_value=1, max_value=6, step=1)
        positions = []
        for i in range(int(num_legs)):
            st.markdown(f"**第 {i+1} 腿**")
            cols = st.columns(5)
            with cols[0]:
                leg_type = st.selectbox("类型", ["call", "put"], key=f"leg_type_{i}")
            with cols[1]:
                leg_side = st.selectbox("方向", ["buy (买)", "sell (卖)"], key=f"leg_side_{i}")
            with cols[2]:
                leg_K = st.number_input("行权价", value=float(round(S)), step=1.0, key=f"leg_K_{i}")
            with cols[3]:
                leg_qty = st.number_input("手数", value=1, min_value=1, step=1, key=f"leg_qty_{i}")
            with cols[4]:
                leg_res = black_scholes(S, leg_K, T4, r, q, sigma4, leg_type)
                leg_premium = st.number_input("权利金", value=round(leg_res.price, 2),
                                               step=0.1, key=f"leg_prem_{i}")

            positions.append({
                "type": leg_type,
                "strike": leg_K,
                "premium": leg_premium,
                "qty": int(leg_qty),
                "side": "buy" if "buy" in leg_side else "sell",
            })

    # 到期盈亏图
    if positions:
        st.markdown("---")
        st.subheader("到期盈亏图")
        all_strikes = [p["strike"] for p in positions]
        s_min = min(all_strikes) * 0.85
        s_max = max(all_strikes) * 1.15
        S_pnl = np.linspace(s_min, s_max, 500)
        pnl = option_pnl_at_expiry(S_pnl, positions)

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=S_pnl, y=pnl, fill="tozeroy",
            line=dict(color="#2196F3", width=2),
            fillcolor="rgba(76,175,80,0.15)",
            name="盈亏 (每手×100股)"
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_pnl.add_vline(x=S, line_dash="dot", line_color="red",
                          annotation_text=f"现价 ${S:.0f}")
        fig_pnl.update_layout(
            height=450, template="plotly_white",
            xaxis_title="到期时标的价格 ($)",
            yaxis_title="盈亏 ($)",
            title="到期盈亏图 (P&L at Expiration)"
        )
        st.plotly_chart(fig_pnl, use_container_width=True)


# =============================================
# Tab 5: 期权入门指南
# =============================================
with tab5:
    st.subheader("期权定价核心概念")

    st.markdown("""
    ### 什么是期权？

    期权是一种金融衍生品合约，赋予买方在特定日期前以特定价格买入或卖出标的资产的**权利**（非义务）。

    | 概念 | 说明 |
    |------|------|
    | **Call（看涨期权）** | 买方有权在到期时以行权价**买入**标的股票 |
    | **Put（看跌期权）** | 买方有权在到期时以行权价**卖出**标的股票 |
    | **行权价 (Strike)** | 合约约定的买卖价格 |
    | **权利金 (Premium)** | 期权的市场价格，即你需要付出的成本 |
    | **到期日 (Expiration)** | 期权合约到期的日期 |
    | **1手 = 100股** | 美股期权每手合约对应100股标的股票 |

    ---

    ### 内在价值 vs 时间价值

    **期权价格 = 内在价值 + 时间价值**

    - **内在价值**: 立即行权可获得的收益。Call: max(S-K, 0)；Put: max(K-S, 0)
    - **时间价值**: 到期前股价可能变动带来的潜在收益。随到期日临近而衰减（Theta衰减）

    | 状态 | Call | Put | 说明 |
    |------|------|-----|------|
    | **ITM (实值)** | S > K | S < K | 有内在价值 |
    | **ATM (平值)** | S ≈ K | S ≈ K | 内在价值 ≈ 0 |
    | **OTM (虚值)** | S < K | S > K | 无内在价值 |

    ---

    ### Greeks 解读

    | Greek | 含义 | 实用建议 |
    |-------|------|----------|
    | **Delta (Δ)** | 股价变$1 → 期权变多少 | Call: 0~1, Put: -1~0。Delta ≈ 到期时ITM的概率 |
    | **Gamma (Γ)** | 股价变$1 → Delta变多少 | ATM期权Gamma最大，越近到期Gamma越尖锐 |
    | **Theta (Θ)** | 每过1天 → 期权贬值多少 | 买方的敌人、卖方的朋友。ATM衰减最快 |
    | **Vega (ν)** | 波动率升1% → 期权变多少 | 财报前波动率膨胀(IV crush)是关键考量 |
    | **Rho (ρ)** | 利率升1% → 期权变多少 | 通常影响较小，长期期权更敏感 |

    ---

    ### Black-Scholes 模型

    最经典的期权定价模型，假设：
    1. 标的价格服从几何布朗运动（对数正态分布）
    2. 无交易成本、可连续交易
    3. 波动率恒定
    4. 无风险利率恒定

    **核心公式 (Call):**

    $C = S·e^{-qT}·N(d_1) - K·e^{-rT}·N(d_2)$

    其中 $d_1 = \\frac{\\ln(S/K) + (r-q+\\sigma^2/2)T}{\\sigma\\sqrt{T}}$, $d_2 = d_1 - \\sigma\\sqrt{T}$

    ---

    ### 隐含波动率 (IV)

    市场价格反推出的波动率。**IV 是市场对未来波动的预期。**

    - IV 高 → 期权贵（市场预期大波动，如财报前）
    - IV 低 → 期权便宜
    - **IV Crush**: 财报公布后IV急剧下降，即使方向判断正确也可能亏钱

    ---

    ### 常用策略速查

    | 策略 | 适用场景 | 最大盈利 | 最大亏损 |
    |------|----------|----------|----------|
    | **买Call** | 强烈看涨 | 无限 | 权利金 |
    | **买Put** | 强烈看跌 | K - 权利金 | 权利金 |
    | **Covered Call** | 温和看涨/持股增收 | 行权价-买入价+权利金 | 股价下跌 |
    | **牛市价差** | 温和看涨 | K2-K1-净权利金 | 净权利金 |
    | **熊市价差** | 温和看跌 | K2-K1-净权利金 | 净权利金 |
    | **跨式 (Straddle)** | 预期大幅波动 | 无限 | 总权利金 |
    | **铁鹰 (Iron Condor)** | 预期窄幅震荡 | 净收入权利金 | 翅膀宽度-权利金 |

    ---

    ### 实操建议

    1. **先学会看 Greeks**: 尤其是 Delta 和 Theta，它们最直接影响你的盈亏
    2. **注意 IV 水平**: 不要在高 IV 时买期权（贵），也不要在低 IV 时卖期权（便宜）
    3. **控制仓位**: 单笔期权交易不超过总资金的 2-5%
    4. **优先使用价差策略**: 限制了最大亏损，比裸买/裸卖更安全
    5. **关注流动性**: 选择成交量大、买卖价差小的合约
    """)
