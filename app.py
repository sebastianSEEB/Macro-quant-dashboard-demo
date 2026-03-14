import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta, datetime

# ==========================================
# 1. MASTER DATA & STATE LAYER
# ==========================================
MASTER_TICKERS = {
    "EQNR.OL": "Equinor", "DNB.OL": "DNB", "KOG.OL": "Kongsberg Gruppen", "TEL.OL": "Telenor", 
    "NHY.OL": "Norsk Hydro", "YAR.OL": "Yara International", "GJF.OL": "Gjensidige Forsikring", 
    "MOWI.OL": "Mowi", "ORK.OL": "Orkla", "AKRBP.OL": "Aker BP", "SUBC.OL": "Subsea 7", 
    "TOM.OL": "Tomra Systems", "SALM.OL": "SalMar", "STB.OL": "Storebrand", "SCHA.OL": "Schibsted A", 
    "NEL.OL": "Nel", "BAKKA.OL": "Bakkafrost", "FRO.OL": "Frontline", "VAR.OL": "Vår Energi", 
    "ELK.OL": "Elkem", "PGS.OL": "PGS", "RENA.OL": "Rana Gruber", "NAS.OL": "Norwegian Air Shuttle", 
    "ABL.OL": "Abl Group", "VISTN.OL": "Vistin Pharma", "MPCC.OL": "MPC Container Ships",
    "SNI.OL": "Stolt-Nielsen", "KAHOT.OL": "Kahoot!", "NOD.OL": "Nordic Semiconductor",
    "ENTRA.OL": "Entra", "AFG.OL": "AF Gruppen", "BWLPG.OL": "BW LPG", "GIG.OL": "Gaming Innovation Group",
    "VEI.OL": "Veidekke", "BWE.OL": "BW Energy", "LSG.OL": "Lerøy Seafood", "PROT.OL": "Protector Forsikring",
    "GOGL.OL": "Golden Ocean", "BOUV.OL": "Bouvet", "KCC.OL": "Klaveness Combination Carriers",
    "KIT.OL": "Kitron", "SCATC.OL": "Scatec", "AUSS.OL": "Austevoll Seafood", "HEX.OL": "Hexagon Composites",
    "VOLV-B.ST": "Volvo (Nordic)", "FLEX.OL": "Flex LNG", "OET.OL": "Odfjell Drilling",
    "AKVA.OL": "AKVA Group", "BGBIO.OL": "BerGenBio", "BORG.OL": "Borregaard", "CADLR.OL": "Cadeler"
}

# Macroeconomic indicators to pull from Yahoo Finance
MACRO_TICKERS = {
    "^TNX": "10Y_Yield",
    "^VIX": "VIX",
    "BZ=F": "Brent_Crude",
    "NOK=X": "USD_NOK"
}

horizons = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "1y": 252}

def initialize_state():
    if 'known_tickers' not in st.session_state:
        st.session_state.known_tickers = MASTER_TICKERS.copy()
    if 'active_tickers' not in st.session_state:
        st.session_state.active_tickers = list(MASTER_TICKERS.keys())[:10]
    if 'ms_tickers' not in st.session_state:
        st.session_state.ms_tickers = st.session_state.active_tickers
        
    # Default variables for the regression toggle table
    if 'regression_features' not in st.session_state:
        st.session_state.regression_features = pd.DataFrame({
            'Feature': ['Momentum (Past 1D Return)', 'Brent Crude Change', '10Y Yield Change', 'VIX Change', 'USD/NOK Change'],
            'Column_Name': ['ret_1d', 'Brent_Crude_ret', '10Y_Yield_diff', 'VIX_ret', 'USD_NOK_ret'],
            'Include': [True, True, False, True, False]
        })

def sync_multiselect():
    st.session_state.active_tickers = st.session_state.ms_tickers

def add_top_x(x: int):
    top_x = list(MASTER_TICKERS.keys())[:x]
    current_active = set(st.session_state.active_tickers)
    for t in top_x:
        if t not in current_active:
            st.session_state.active_tickers.append(t)
    st.session_state.ms_tickers = st.session_state.active_tickers

def add_all_tickers():
    st.session_state.active_tickers = list(MASTER_TICKERS.keys())
    st.session_state.ms_tickers = st.session_state.active_tickers

def remove_all_tickers():
    st.session_state.active_tickers = []
    st.session_state.ms_tickers = []

# ==========================================
# 2. DATA LAYER (YAHOO FINANCE)
# ==========================================
@st.cache_data(ttl=3600) 
def load_raw_data(active_tickers: list) -> pd.DataFrame:
    if not active_tickers:
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_tickers_to_fetch = active_tickers + list(MACRO_TICKERS.keys())
    
    with st.spinner(f'Fetching Data & Macro Indicators...'):
        try:
            data = yf.download(all_tickers_to_fetch, start=start_date, end=end_date, group_by='ticker', progress=False)
            
            macro_df = pd.DataFrame(index=data.index)
            for mticker, mname in MACRO_TICKERS.items():
                if mticker in data:
                    if mname == "10Y_Yield":
                        macro_df[f'{mname}_diff'] = data[mticker]['Close'].diff()
                    else:
                        macro_df[f'{mname}_ret'] = data[mticker]['Close'].pct_change()
            
            macro_df = macro_df.reset_index().rename(columns={'Date': 'date'})
            macro_df['date'] = pd.to_datetime(macro_df['date'])

            stock_data = []
            for ticker in active_tickers:
                if ticker in data:
                    ticker_df = data[ticker][['Close']].reset_index()
                    ticker_df.columns = ['date', 'close']
                    ticker_df['ticker'] = ticker
                    ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                    stock_data.append(ticker_df)
            
            if not stock_data:
                return pd.DataFrame()
                
            final_stock_df = pd.concat(stock_data, ignore_index=True)
            merged_df = pd.merge(final_stock_df, macro_df, on='date', how='left')
            return merged_df
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.sort_values(by=['ticker', 'date']).drop_duplicates(subset=['ticker', 'date'])
    return df.ffill() 

# ==========================================
# 3. QUANTITATIVE ANALYSIS LAYER
# ==========================================
@st.cache_data
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = clean_and_validate(df.copy())
    
    for h_name, h_days in horizons.items():
        df[f'ret_{h_name}'] = df.groupby('ticker')['close'].pct_change(periods=h_days)
        
    df['next_ret_1d'] = df.groupby('ticker')['ret_1d'].shift(-1)
    return df

def run_multivariate_regression(df: pd.DataFrame, features: list):
    if df.empty or not features: 
        return pd.DataFrame(), pd.DataFrame()
        
    analysis_df = df.dropna(subset=['next_ret_1d'] + features).copy()
    if analysis_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    Y = analysis_df['next_ret_1d']
    X = sm.add_constant(analysis_df[features])
    
    # 1. OLS Regression
    model = sm.OLS(Y, X).fit(cov_type='HC1')
    ols_results = pd.DataFrame({
        'Coefficient (β)': model.params,
        't-Statistic': model.tvalues,
        'p-Value': model.pvalues
    }).drop('const', errors='ignore')
    
    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(analysis_df[features], Y)
    rf_results = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return ols_results, rf_results

def analyze_individual_momentum(df: pd.DataFrame, target_horizon: str = 'ret_1d') -> list:
    if df.empty: return []
    winning_tickers = []
    analysis_df = df.dropna(subset=['next_ret_1d', target_horizon])
    
    for ticker, group in analysis_df.groupby('ticker'):
        if len(group) < 30: continue
        Y = group['next_ret_1d']
        X = sm.add_constant(group[target_horizon])
        try:
            model = sm.OLS(Y, X).fit(cov_type='HC1')
            if target_horizon in model.params:
                if model.params[target_horizon] > 0 and model.pvalues[target_horizon] < 0.05:
                    winning_tickers.append(ticker)
        except Exception:
            pass
    return winning_tickers

# ==========================================
# 4. DASHBOARD & PRESENTATION LAYER
# ==========================================
def style_returns(val):
    if pd.isna(val) or isinstance(val, bool): return ''
    color = '#2ECC71' if val > 0 else '#E74C3C' if val < 0 else 'white'
    return f'color: {color}'

def main():
    st.set_page_config(page_title="Macro-Quant Forecasting Dashboard", layout="wide")
    initialize_state()
    
    # --- SIDEBAR UI ---
    st.sidebar.header("⚙️ Portfolio Management")
    
    new_ticker = st.sidebar.text_input("Add Custom Ticker (e.g., AAPL)")
    if st.sidebar.button("Add Single Ticker") and new_ticker:
        new_ticker = new_ticker.upper().strip()
        if new_ticker not in st.session_state.known_tickers:
            st.session_state.known_tickers[new_ticker] = new_ticker
        if new_ticker not in st.session_state.active_tickers:
            st.session_state.active_tickers.append(new_ticker)
            st.session_state.ms_tickers = st.session_state.active_tickers
        st.rerun()
        
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Bulk Actions")
    col_bulk1, col_bulk2 = st.sidebar.columns(2)
    if col_bulk1.button("✅ Add NO Top"): add_all_tickers(); st.rerun()
    if col_bulk2.button("❌ Clear All"): remove_all_tickers(); st.rerun()
    
    st.sidebar.markdown("---")
    
    # Fixed Sidebar Buttons Layout (3 Columns to prevent squishing)
    st.sidebar.subheader("Add Top Norwegian Stocks")
    top_stocks_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cols = st.sidebar.columns(3)
    
    for i, val in enumerate(top_stocks_vals):
        # i % 3 places it in col 0, 1, or 2 repeatedly 
        if cols[i % 3].button(f"+{val}", key=f"btn_{val}", use_container_width=True):
            add_top_x(val)
            st.rerun()

    st.sidebar.markdown("---")
    
    st.sidebar.multiselect(
        "Active Tickers (Deselect to Hide)",
        options=list(st.session_state.known_tickers.keys()),
        key="ms_tickers",
        on_change=sync_multiselect
    )
    
    # --- MAIN DASHBOARD ---
    st.title("🌍 Macro-Quant Predictive Engine")
    st.caption("Multivariate Regression & Machine Learning models powered by Yahoo Finance")
    
    if not st.session_state.active_tickers:
        st.warning("⚠️ Please select at least one ticker.")
        st.stop()
    
    raw_df = load_raw_data(st.session_state.active_tickers)
    df = compute_features(raw_df)
    
    if df.empty:
        st.error("No data fetched.")
        st.stop()
        
    latest_date = df['date'].max()
    
    col_table, col_chart = st.columns([1.7, 2])
    
    with col_table:
        st.subheader("Current Snapshot")
        latest_df = df[df['date'] == latest_date].copy()
        latest_df['Company'] = latest_df['ticker'].map(st.session_state.known_tickers)
        
        cols_to_show = ['ticker', 'Company', 'close', 'ret_1d', 'ret_1w', 'ret_1m', 'ret_3m', 'ret_1y']
        table_df = latest_df[cols_to_show].set_index('ticker')
        table_df.insert(0, '🗑️ Remove', False)
        
        styled_table = table_df.style.format({
            'close': "{:.2f}", 'ret_1d': "{:.2%}", 'ret_1w': "{:.2%}",
            'ret_1m': "{:.2%}", 'ret_3m': "{:.2%}", 'ret_1y': "{:.2%}"
        }).map(style_returns, subset=['ret_1d', 'ret_1w', 'ret_1m', 'ret_3m', 'ret_1y'])
        
        edited_df = st.data_editor(
            styled_table, width="stretch", height=400,
            disabled=['Company', 'close', 'ret_1d', 'ret_1w', 'ret_1m', 'ret_3m', 'ret_1y'], 
            column_config={
                "🗑️ Remove": st.column_config.CheckboxColumn("Remove", default=False),
                "Company": st.column_config.TextColumn("Company", width="large")
            }
        )
        
        tickers_to_remove = edited_df[edited_df['🗑️ Remove'] == True].index.tolist()
        if tickers_to_remove:
            for t in tickers_to_remove:
                if t in st.session_state.active_tickers:
                    st.session_state.active_tickers.remove(t)
            st.session_state.ms_tickers = st.session_state.active_tickers
            st.rerun()

    with col_chart:
        st.subheader("Historical Performance")
        chart_df = df.copy()
        first_prices = chart_df.groupby('ticker')['close'].transform('first')
        chart_df['Normalized'] = (chart_df['close'] / first_prices) * 100
            
        fig = px.line(chart_df, x='date', y='Normalized', color='ticker', template="plotly_dark")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ---------------------------------------------------------
    # PART 2: MULTIVARIATE PREDICTIVE ENGINE
    # ---------------------------------------------------------
    st.header("🔬 Multivariate Predictive Engine")
    st.write("Configure the independent variables to test which macroeconomic factors drive next-day returns for your selected portfolio.")
    
    eng_col1, eng_col2 = st.columns([1, 2])
    
    with eng_col1:
        st.subheader("1. Select Variables")
        st.write("Check the variables to include in the models:")
        
        edited_features = st.data_editor(
            st.session_state.regression_features,
            hide_index=True,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include in Model", default=False),
                "Column_Name": None
            },
            disabled=["Feature"]
        )
        st.session_state.regression_features = edited_features
        
        active_features_df = edited_features[edited_features['Include'] == True]
        active_cols = active_features_df['Column_Name'].tolist()

    with eng_col2:
        st.subheader("2. Model Results")
        
        if len(active_cols) == 0:
            st.warning("Please select at least one variable to run the models.")
        else:
            ols_res, rf_res = run_multivariate_regression(df, active_cols)
            
            if ols_res.empty:
                st.error("Not enough overlapping data to run the regression. Try selecting fewer variables.")
            else:
                tabs = st.tabs(["📊 OLS Regression (Linear)", "🤖 Random Forest (Non-Linear)"])
                name_mapping = dict(zip(active_features_df['Column_Name'], active_features_df['Feature']))
                
                # Tab 1: Econometrics
                with tabs[0]:
                    st.markdown(r"**Equation:** $R_{t+1} = \beta_0 + \beta_1 X_1 + ... + \beta_k X_k + u_t$")
                    ols_res.index = ols_res.index.map(name_mapping)
                    
                    def highlight_pvals(val):
                        return 'background-color: rgba(46, 204, 113, 0.2)' if val < 0.05 else ''
                    
                    st.dataframe(
                        ols_res.style.format({'Coefficient (β)': '{:.5f}', 't-Statistic': '{:.2f}', 'p-Value': '{:.4f}'})
                        .map(highlight_pvals, subset=['p-Value']),
                        use_container_width=True
                    )
                    
                    # DYNAMIC OLS TEXT EXPLAINER
                    sig_df = ols_res[ols_res['p-Value'] < 0.05]
                    
                    if not sig_df.empty:
                        pos_vars = sig_df[sig_df['Coefficient (β)'] > 0].index.tolist()
                        neg_vars = sig_df[sig_df['Coefficient (β)'] < 0].index.tolist()
                        
                        explanation = "**📈 Econometric Findings:**\n\nBased on rigorous OLS regression (using robust standard errors), we found statistically significant linear relationships ($p < 0.05$):\n"
                        if pos_vars:
                            explanation += f"* **Positive Drivers:** {', '.join(pos_vars)}. When these increase, the next-day stock return reliably tends to go **up**.\n"
                        if neg_vars:
                            explanation += f"* **Negative Drivers:** {', '.join(neg_vars)}. When these increase, the next-day stock return reliably tends to go **down**.\n"
                        
                        st.success(explanation)
                    else:
                        st.info("**📈 Econometric Findings:**\n\nCurrently, none of the selected variables show a statistically significant linear relationship ($p < 0.05$) with next-day returns. The model suggests these specific factors are not reliable independent predictors for this specific portfolio right now.")
                
                # Tab 2: Machine Learning
                with tabs[1]:
                    rf_res['Feature'] = rf_res['Feature'].map(name_mapping)
                    
                    fig_rf = px.bar(
                        rf_res, x='Importance', y='Feature', orientation='h',
                        title="Random Forest Feature Importance",
                        template="plotly_dark",
                        color='Importance', color_continuous_scale='viridis'
                    )
                    fig_rf.update_layout(margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
                    st.plotly_chart(fig_rf, use_container_width=True)
                    
                    if not rf_res.empty:
                        top_feature = rf_res.iloc[-1]['Feature']
                        top_score = rf_res.iloc[-1]['Importance']
                        
                        if len(rf_res) >= 2:
                            runner_up = rf_res.iloc[-2]['Feature']
                            st.info(f"""
                            **🤖 AI Model Insight:**
                            Based on the current data, the Random Forest algorithm has identified **{top_feature}** as the single most dominant factor driving next-day returns for this portfolio, carrying **{top_score:.1%}** of the predictive weight. 
                            
                            This means that right now, the non-linear movements in {top_feature} (followed closely by {runner_up}) are overriding other macroeconomic indicators. If you are predicting tomorrow's price action, {top_feature} is the primary signal to watch.
                            """)
                        else:
                            st.info(f"**🤖 AI Model Insight:** You currently only have **{top_feature}** selected, accounting for 100% of the tested variance. Add more features to see how they compete for predictive importance.")

    st.markdown("---")
    
    # ---------------------------------------------------------
    # PART 3: SINGLE STOCK SPOTLIGHT
    # ---------------------------------------------------------
    st.subheader("🎯 Single Stock Spotlight: Individual Momentum Winners")
    with st.spinner("Running individual continuous momentum regressions..."):
        individual_winners = analyze_individual_momentum(df, target_horizon='ret_1d')
        
    if individual_winners:
        winner_names = [st.session_state.known_tickers.get(t, t) for t in individual_winners]
        st.success(f"**Actionable Insight:** We found individual stocks with a **statistically significant continuous momentum effect** (A higher return today reliably predicts a higher return tomorrow). \n\n**Winners:** {', '.join(winner_names)}.")
    else:
        st.info("**Actionable Insight:** Currently, none of the active individual stocks show a statistically significant positive momentum effect from the previous day.")

if __name__ == "__main__":
    main()
