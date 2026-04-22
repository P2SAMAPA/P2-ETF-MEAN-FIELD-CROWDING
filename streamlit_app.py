"""
Streamlit Dashboard for Mean-Field Crowding Engine with Advanced Features.
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
    .valid-positive { color: #28a745; }
    .valid-negative { color: #dc3545; }
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

def valid_badge(val):
    if val > 0:
        return f'<span class="valid-positive">+{val:.2f}</span>'
    else:
        return f'<span class="valid-negative">{val:.2f}</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧑‍🤝‍🧑 P2Quant Mean-Field Crowding</div>', unsafe_allow_html=True)
st.markdown('<div>Advanced Crowding Analysis – Kalman, Decomposition, Predictive Validation</div>', unsafe_allow_html=True)

with st.expander("📘 What does 'Crowding' mean?", expanded=False):
    st.markdown("""
    ### Crowding Score (0–1)
    
    **Crowding** measures how "popular" an ETF is, using:
    - **Momentum Crowding**: Extreme recent returns.
    - **Volume Crowding**: Elevated trading volume.
    - **Macro Crowding**: Dynamic sensitivity to VIX (Kalman filter).
    
    **Advanced Features:**
    - **Cross‑sectional rank**: Scores normalized within universe.
    - **Crowding momentum**: Rate of change — rising crowding may signal imminent reversal.
    - **Volume‑weighted macro**: Higher volume amplifies macro sensitivity.
    - **Regime‑adjusted**: Thresholds adapt to VIX level.
    - **Return decomposition**: Adj Return = Pure Alpha – Crowding Penalty.
    - **Predictive validity**: Historical correlation between crowding and forward returns.
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
            mom = pick.get('crowding_momentum', 0.0)
            alpha = pick.get('alpha', ret)
            penalty = pick.get('crowding_penalty', 0.0)
            valid = pick.get('predictive_validity', 0.0)
            mom_str = f"{mom:+.2f}" if mom else "0.00"
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🧑‍🤝‍🧑 TOP PICK (Least Crowded)</div>
                <div class="hero-ticker">{ticker}</div>
                <div style="font-size: 1.5rem;">Adj Return: {ret*100:.2f}%</div>
                <div style="margin-top: 1rem;">Crowding: {crowding_badge(crowd)} (95% CI: {ci_low:.2f}–{ci_high:.2f})</div>
                <div>Crowding Momentum: {mom_str} | Alpha: {alpha*100:.2f}% | Penalty: {penalty*100:.2f}%</div>
                <div>Predictive Validity: {valid_badge(valid)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Adj Return": f"{p['expected_return_adj']*100:.2f}%",
                    "Crowding": f"{p['crowding_score']:.2f}",
                    "Momentum": f"{p.get('crowding_momentum', 0.0):+.2f}",
                    "Alpha": f"{p.get('alpha', 0.0)*100:.2f}%",
                    "Penalty": f"{p.get('crowding_penalty', 0.0)*100:.2f}%",
                    "Predictive": f"{p.get('predictive_validity', 0.0):+.2f}"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Raw Return": f"{d['expected_return_raw']*100:.2f}%",
                    "Crowding": f"{d['crowding_score']:.2f}",
                    "Momentum": f"{d.get('crowding_momentum', 0.0):+.2f}",
                    "Alpha": f"{d.get('alpha', 0.0)*100:.2f}%",
                    "Penalty": f"{d.get('crowding_penalty', 0.0)*100:.2f}%",
                    "Adj Return": f"{d['expected_return_adj']*100:.2f}%",
                    "Predictive": f"{d.get('predictive_validity', 0.0):+.2f}"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Adj Return", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
