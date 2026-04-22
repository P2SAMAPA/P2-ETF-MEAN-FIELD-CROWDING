"""
Streamlit Dashboard for Mean-Field Crowding Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Mean-Field Crowding", page_icon="🧑‍🤝‍🧑", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .crowding-high { color: #dc3545; font-weight: 600; }
    .crowding-mid { color: #ffc107; font-weight: 600; }
    .crowding-low { color: #28a745; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("mean_field_crowding_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def crowding_badge(score):
    if score > 0.7:
        return f'<span class="crowding-high">High ({score:.2f})</span>'
    elif score > 0.4:
        return f'<span class="crowding-mid">Mid ({score:.2f})</span>'
    else:
        return f'<span class="crowding-low">Low ({score:.2f})</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧑‍🤝‍🧑 P2Quant Mean-Field Crowding</div>', unsafe_allow_html=True)
st.markdown('<div>Mean‑Field Crowding Proxy – Momentum, Volume & Macro Sensitivity</div>', unsafe_allow_html=True)

# --- Explanation Expander ---
with st.expander("📘 What does 'Crowding' mean?", expanded=False):
    st.markdown("""
    ### Crowding Score (0–1)
    
    **Crowding** measures how "popular" or "overbought" an ETF is, based on three signals:
    
    1. **Momentum Crowding**: How extreme recent returns are relative to history.  
       *High z‑score → potential mean reversion risk.*
    2. **Volume Crowding**: Recent trading volume relative to long‑term average.  
       *Elevated volume often indicates institutional positioning.*
    3. **Macro Crowding**: Correlation of the ETF's returns with the VIX (fear index).  
       *High correlation means the ETF is sensitive to macro shocks and likely crowded by systematic strategies.*
    
    **Interpretation:**
    - **High (>0.7)** : Strong crowding signal — risk of reversal if momentum fades.
    - **Mid (0.4–0.7)** : Moderate crowding — neutral.
    - **Low (<0.4)** : Little crowding — less downside risk from positioning.
    
    **Crowding‑Adjusted Return:**  
    `Adj Return = Raw Return × (1 - Crowding Score)`  
    *(Negative raw returns get a smaller penalty, as short‑side crowding is less risky.)*
    
    The engine selects ETFs with the **highest crowding‑adjusted return**, favoring assets with positive return potential and low crowding risk.
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            ret = pick['expected_return_adj']
            crowd = pick['crowding_score']
            ci_low = pick.get('crowding_ci_lower', crowd)
            ci_high = pick.get('crowding_ci_upper', crowd)
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🧑‍🤝‍🧑 TOP PICK (Least Crowded)</div>
                <div class="hero-ticker">{ticker}</div>
                <div style="font-size: 1.5rem;">Adj Return: {ret*100:.2f}%</div>
                <div style="margin-top: 1rem;">Crowding: {crowding_badge(crowd)} (95% CI: {ci_low:.2f}–{ci_high:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Adj Return": f"{p['expected_return_adj']*100:.2f}%",
                    "Crowding": f"{p['crowding_score']:.2f}",
                    "CI Low": f"{p.get('crowding_ci_lower', 0):.2f}",
                    "CI High": f"{p.get('crowding_ci_upper', 0):.2f}"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Raw Return": f"{d['expected_return_raw']*100:.2f}%",
                    "Crowding Score": f"{d['crowding_score']:.2f}",
                    "Adj Return": f"{d['expected_return_adj']*100:.2f}%"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Adj Return", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
