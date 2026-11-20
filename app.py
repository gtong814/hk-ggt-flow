import streamlit as st
import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="港股通成份股資金流掃描器", layout="wide")

st.title("🇭🇰 港股通成份股 — 主動買賣資金流 Dashboard")
st.caption("自動獲取港股通名單，批量計算收市價位置 (MFM)、成交額加權資金流與量比 (RVOL)，篩選機構吸籌與派貨異動股")

# === 備用核心港股通名單（以防外部名單接口暫時繁忙） ===
FALLBACK_STOCKS = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W", "1810.HK": "小米集團-W",
    "0981.HK": "中芯國際", "0939.HK": "建設銀行", "0941.HK": "中國移動", "0883.HK": "中國海洋石油",
    "1211.HK": "比亞迪股份", "2015.HK": "理想汽車-W", "9868.HK": "小鵬汽車-W", "9660.HK": "地平線機器人-W",
    "1133.HK": "哈爾濱電氣", "1686.HK": "新意網集團", "2858.HK": "易鑫集團", "0388.HK": "香港交易所",
    "2318.HK": "中國平安", "1299.HK": "友邦保險", "0005.HK": "匯豐控股", "3968.HK": "招商銀行",
    "1398.HK": "工商銀行", "3988.HK": "中國銀行", "0857.HK": "中國石油股份", "1088.HK": "中國神華",
    "2899.HK": "紫金礦業", "9999.HK": "網易-S", "9618.HK": "京東集團-SW", "1024.HK": "快手-W",
    "2269.HK": "藥明生物", "1801.HK": "信達生物", "6160.HK": "百濟神州", "2382.HK": "舜宇光學科技",
    "0285.HK": "比亞迪電子", "0992.HK": "聯想集團", "2020.HK": "安踏體育", "2331.HK": "李寧",
    "9992.HK": "泡泡瑪特", "1919.HK": "中遠海控", "0728.HK": "中國電信", "0762.HK": "中國聯通",
    "2800.HK": "盈富基金", "3033.HK": "南方恒生科技"
}

@st.cache_data(ttl=86400)
def get_ggt_list():
    """經 akshare 自動抓取最新港股通成份股名單"""
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        # 轉換為 yfinance 4位數格式 (例如 00700 -> 0700.HK)
        stock_map = {}
        for _, row in df_ggt.iterrows():
            code_raw = str(row["代码"]).strip()
            name = str(row["名称"]).strip()
            # 港股通5位數代碼去掉最前面一個0變成4位數，若原本5位不以0開頭則保留
            code_4 = code_raw[1:] if len(code_raw) == 5 and code_raw.startswith("0") else code_raw.zfill(4)
            stock_map[f"{code_4}.HK"] = name
        if len(stock_map) > 50:
            return stock_map, True
    except Exception:
        pass
    return FALLBACK_STOCKS, False

# === 側邊欄設定 ===
with st.sidebar:
    st.header("⚙️ 掃描設定")
    scan_mode = st.radio(
        "掃描範圍",
        ["⚡ 熱門核心港股通 (42隻，約5秒)", "🌐 全量港股通成份股 (約500隻，約1分鐘)"],
        index=0
    )
    
    st.divider()
    len_fast = st.slider("短線資金流週期 (日)", 3, 15, 5)
    len_slow = st.slider("中線資金流週期 (日)", 10, 60, 20)
    rvol_len = st.slider("量比基準週期 (日)", 10, 60, 20)
    rvol_thres = st.slider("大資金異動量比門檻", 1.0, 3.0, 1.5, 0.1)
    min_turnover_m = st.number_input("最低日均成交額過濾 (百萬港元)", min_value=0, value=20, step=10)
    
    if st.button("🔄 清除快取並重新掃描", use_container_width=True):
        st.cache_data.clear()

# 決定要掃描嘅名單
if "全量" in scan_mode:
    stock_map, is_live = get_ggt_list()
    if is_live:
        st.sidebar.success(f"已同步最新港股通名單：共 {len(stock_map)} 隻")
    else:
        st.sidebar.warning("外部名單接口繁忙，暫改用內置核心名單。")
else:
    stock_map = FALLBACK_STOCKS

# === 分批下載與計算函數 ===
@st.cache_data(ttl=600)
def batch_scan_stocks(s_map, fast_n, slow_n, rvol_n, rvol_th, min_turn_m):
    tickers = list(s_map.keys())
    chunk_size = 60
    results = []
    
    progress_bar = st.progress(0, text="正在分批下載港股通價量數據...")
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    
    for idx in range(total_chunks):
        chunk = tickers[idx * chunk_size : (idx + 1) * chunk_size]
        progress_bar.progress((idx + 1) / total_chunks, text=f"掃描進度：{min((idx + 1) * chunk_size, len(tickers))} / {len(tickers)} 隻...")
        
        try:
            data = yf.download(
                chunk, period="3mo", interval="1d",
                group_by="ticker", threads=True, progress=False, auto_adjust=False
            )
        except Exception:
            continue
            
        for sym in chunk:
            try:
                df = data[sym] if len(chunk) > 1 else data
                df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
                if len(df) < max(slow_n, rvol_n) + 2:
                    continue
                
                # 成交額過濾（剔除成交極低嘅冷門股）
                df["Turnover"] = df["Close"] * df["Volume"]
                avg_turn_m = df["Turnover"].tail(5).mean() / 1_000_000
                if avg_turn_m < min_turn_m:
                    continue
                
                # 1. 主動買賣乘數 (MFM)
                hl_range = df["High"] - df["Low"]
                df["MFM"] = np.where(
                    hl_range != 0,
                    ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range,
                    np.where(df["Close"] > df["Open"], 1.0, np.where(df["Close"] < df["Open"], -1.0, 0.0))
                )
                
                # 2. 淨資金流與快慢線
                df["NetFlow"] = df["MFM"] * df["Turnover"]
                fast_turn = df["Turnover"].rolling(fast_n).sum()
                slow_turn = df["Turnover"].rolling(slow_n).sum()
                df["FastFlowPct"] = np.where(fast_turn != 0, (df["NetFlow"].rolling(fast_n).sum() / fast_turn) * 100, 0.0)
                df["SlowFlowPct"] = np.where(slow_turn != 0, (df["NetFlow"].rolling(slow_n).sum() / slow_turn) * 100, 0.0)
                
                # 3. 量比與升跌
                avg_vol = df["Volume"].rolling(rvol_n).mean()
                df["RVOL"] = np.where(avg_vol != 0, df["Volume"] / avg_vol, 1.0)
                df["ChgPct"] = df["Close"].pct_change() * 100
                
                last = df.iloc[-1]
                hi_10 = df["Close"].tail(10).max()
                lo_10 = df["Close"].tail(10).min()
                
                mfm_pct = float(last["MFM"] * 100)
                f_pct = float(last["FastFlowPct"])
                s_pct = float(last["SlowFlowPct"])
                rvol = float(last["RVOL"])
                
                # 主力狀態分類
                if last["Close"] >= hi_10 and s_pct < 0 and mfm_pct < -20:
                    status = "⚠️ 拉高派貨"
                elif last["Close"] <= lo_10 * 1.05 and s_pct > 10 and rvol >= rvol_th:
                    status = "🟢 暗中吸籌"
                elif s_pct >= 15 and mfm_pct > 20 and rvol >= rvol_th:
                    status = "🔥 機構搶貨"
                elif s_pct <= -15 and mfm_pct < -20 and rvol >= rvol_th:
                    status = "🔻 資金離場"
                elif f_pct > 10 and s_pct > 0:
                    status = "↗️ 資金流入"
                elif f_pct < -10 and s_pct < 0:
                    status = "↘️ 資金流出"
                else:
                    status = "⚪ 中性觀望"
                    
                results.append({
                    "代號": sym,
                    "名稱": s_map.get(sym, ""),
                    "現價": round(float(last["Close"]), 2),
                    "升跌(%)": round(float(last["ChgPct"]), 2),
                    "單日力道(%)": round(mfm_pct, 1),
                    f"{fast_n}日流(%)": round(f_pct, 1),
                    f"{slow_n}日流(%)": round(s_pct, 1),
                    "量比(x)": round(rvol, 2),
                    "5日均成交(M)": round(float(avg_turn_m), 1),
                    "主力狀態": status
                })
            except Exception:
                continue
                
    progress_bar.empty()
    return pd.DataFrame(results)

res_df = batch_scan_stocks(stock_map, len_fast, len_slow, rvol_len, rvol_thres, min_turnover_m)

if res_df.empty:
    st.warning("未找到符合條件嘅股票，請調低「最低日均成交額」或稍後再試。")
    st.stop()

# === 頂部市場寬度摘要 ===
c1, c2, c3, c4 = st.columns(4)
c1.metric("合資格掃描股數", f"{len(res_df)} 隻")
c2.metric("🔥 機構搶貨 / 暗中吸籌", f"{res_df['主力狀態'].isin(['🔥 機構搶貨', '🟢 暗中吸籌']).sum()} 隻")
c3.metric("⚠️ 拉高派貨 / 資金離場", f"{res_df['主力狀態'].isin(['⚠️ 拉高派貨', '🔻 資金離場']).sum()} 隻")
c4.metric("資金淨流入比例 (20日>0)", f"{(res_df[f'{len_slow}日流(%)'] > 0).mean() * 100:.1f}%")

# === 狀態篩選器 ===
st.subheader("📋 港股通資金流排行榜")
status_options = ["全部", "🔥 機構搶貨", "🟢 暗中吸籌", "↗️ 資金流入", "⚠️ 拉高派貨", "🔻 資金離場", "↘️ 資金流出", "⚪ 中性觀望"]
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    selected_status = st.multiselect("篩選主力狀態：", status_options, default=["全部"])
with col_f2:
    sort_col = st.selectbox("排序依據：", [f"{len_slow}日流(%)", "單日力道(%)", "量比(x)", "升跌(%)", "5日均成交(M)"])

filtered_df = res_df.copy()
if "全部" not in selected_status and len(selected_status) > 0:
    filtered_df = filtered_df[filtered_df["主力狀態"].isin(selected_status)]

filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)

def highlight_status(val):
    if "搶貨" in val or "吸籌" in val:
        return "background-color: rgba(0, 200, 83, 0.25); color: #00e676; font-weight: bold;"
    elif "派貨" in val or "離場" in val:
        return "background-color: rgba(255, 23, 68, 0.25); color: #ff5252; font-weight: bold;"
    return ""

st.dataframe(
    filtered_df.style.map(highlight_status, subset=["主力狀態"]),
    use_container_width=True,
    hide_index=True,
    height=420
)

# === 個股深度圖表（按需抓取 6 個月完整數據） ===
st.subheader("📈 個股資金流走勢詳情")
if not filtered_df.empty:
    stock_choices = [f"{r['代號']} - {r['名稱']} ({r['主力狀態']})" for _, r in filtered_df.iterrows()]
    chosen_label = st.selectbox("選擇股票查看 K 線與資金流：", stock_choices)
    chosen_sym = chosen_label.split(" - ")[0]
    
    @st.cache_data(ttl=300)
    def get_single_chart_data(sym, fast_n, slow_n):
        df = yf.download(sym, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        hl = df["High"] - df["Low"]
        df["MFM"] = np.where(hl != 0, ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl, 0.0)
        df["Turnover"] = df["Close"] * df["Volume"]
        df["NetFlow"] = df["MFM"] * df["Turnover"]
        df["FastFlowPct"] = (df["NetFlow"].rolling(fast_n).sum() / df["Turnover"].rolling(fast_n).sum()) * 100
        df["SlowFlowPct"] = (df["NetFlow"].rolling(slow_n).sum() / df["Turnover"].rolling(slow_n).sum()) * 100
        return df.tail(90)

    plot_df = get_single_chart_data(chosen_sym, len_fast, len_slow)
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.6, 0.4],
        subplot_titles=(f"{chosen_label} 日線圖", "主動買賣資金流 (%)")
    )
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="K線"
    ), row=1, col=1)
    
    bar_colors = ["#00e676" if v >= 0 else "#ff5252" for v in plot_df["MFM"]]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["MFM"] * 100,
        marker_color=bar_colors, opacity=0.45, name="單日力道(%)"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["FastFlowPct"],
        line=dict(color="#00e5ff", width=2), name=f"{len_fast}日資金流"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["SlowFlowPct"],
        line=dict(color="#ff9100", width=2), name=f"{len_slow}日資金流"
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
