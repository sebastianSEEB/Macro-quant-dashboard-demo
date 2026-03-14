# Macro-quant-dashboard-demo

A full-stack quantitative finance dashboard built with Python and Streamlit. This application fetches real-time global equities and macroeconomic data via Yahoo Finance (`yfinance`) and applies econometric models to identify predictive momentum signals.

**Features**
* **Live Market Data:** Integrates directly with Yahoo Finance for daily pricing of equities and macro indicators (Brent Crude, 10Y Yields, VIX, USD/NOK).
* **Multivariate OLS Regression:** Applies institutional-grade econometrics (with HC1 robust standard errors) to determine if macro factors have a statistically significant linear impact on next-day stock returns.
* **Machine Learning Overlay:** Utilizes a Random Forest Regressor to capture non-linear relationships and ranks economic features by their predictive importance.
* **Single Stock Spotlight:** Automatically loops through the active portfolio to isolate individual assets that exhibit statistically significant momentum effects.
* **Interactive UI:** Built with Streamlit, featuring dynamic data tables, Plotly charts, and customizable macro-variable toggles.

**Installation & Setup**

1. Clone the repository
   ```bash
   git clone [https://github.com/sebastianSEEB/Macro-Quant-Dashboard.git](https://github.com/YOUR_USERNAME/Macro-Quant-Dashboard.git)
   cd Macro-Quant-Dashboard
2. Create a Virtual Environment (Optional but recommended)
   python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install the required dependencies
   pip install -r requirements.txt
4. Run the application
   streamlit run app.py
