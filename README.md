# Energy Optimization (Hackathon Demo)

This project builds an AI/ML-based energy forecasting dashboard using your provided CSVs (simulated IoT).

## What you get
- Appliance-wise **daily** kWh prediction (for a selected `Home ID`)
- Home-wise and building-wise **daily** + **monthly** energy prediction
- Cost prediction: `cost = kWh * tariff`

## Project structure
- `src/preprocess.py` builds daily aggregates into `artifacts/`
- `src/train.py` trains ML models into `models/`
- `dashboard/app.py` is the Streamlit UI

## Run locally
1. Install dependencies
   - `pip install -r requirements.txt`
2. Preprocess (daily aggregation)
   - `python src/preprocess.py`
3. Train models
   - `python src/train.py`
4. Start dashboard
   - `streamlit run dashboard/app.py`

## Notes
- Weather handling:
  - By default the model uses **typical** seasonal outdoor temperature patterns (no leakage of the target day's actual temperature).
  - In the UI you can optionally input “expected outdoor temperature (°C)”.