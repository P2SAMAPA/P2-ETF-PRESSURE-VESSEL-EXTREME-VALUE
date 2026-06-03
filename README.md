# Pressure Vessel – Extreme Value Engine for ETFs

Models ETF returns as pressure fluctuations in a thermodynamic system. Fits a **Weibull distribution** to block maxima (monthly) with **macro variables** (VIX, yields, DXY) as covariates in the scale parameter. The per‑ETF score is the **burst pressure threshold** (99.5% quantile) conditional on current macro state – a measure of extreme tail resilience.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- All available macro variables: VIX, DXY, T10Y2Y, full yield curve
- Block maxima (21‑day rolling) for extreme value analysis
- Weibull log‑likelihood optimisation with macro‑dependent scale
- Score = conditional 99.5% quantile (burst pressure)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-pressure-vessel-extreme-value-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, linear in windows)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High burst pressure → ETF can sustain large positive shocks given current macro → potentially high upside tail.
- Low burst pressure → ETF's tail is limited.

## Requirements

See `requirements.txt`.
