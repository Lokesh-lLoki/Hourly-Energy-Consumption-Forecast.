# Hourly Energy Consumption Forecasting – PJM West

A time-series forecasting project that analyzes historical electricity demand in the PJM West (PJMW) region and produces a 30-day **daily-average demand forecast**. The project combines exploratory data analysis, feature engineering, six forecasting approaches, model evaluation, and an interactive Streamlit dashboard.

> **Forecast granularity:** Although the source dataset contains hourly observations, the supplied 30-day forecast contains 30 daily predictions (one predicted average load per date), not 720 hourly predictions.

## Project overview

Electricity demand changes with the hour of the day, day of the week, and season. This project explores those patterns in PJM West historical load data and compares statistical, machine-learning, and recurrent-neural-network models.

The notebook selects Random Forest for the supplied experiment and generates a recursive 30-day forecast. The Streamlit app presents the forecast, model comparison, demand summaries, and feature importance.

## Dataset

- **Dataset:** PJMW hourly electricity load
- **Source workbook:** `PJMW_MW_Hourly.xlsx`
- **Worksheet:** `PJMW_hourly`
- **Columns:** `Datetime`, `PJMW_MW`
- **Observations:** 143,206 rows in the supplied workbook
- **Date coverage:** 1 April 2002 to 3 August 2018
- **Target:** `PJMW_MW` — electricity demand in megawatts (MW)

The notebook checks duplicate timestamps and irregular hourly intervals, groups duplicate timestamps by their mean, creates a continuous hourly index, and applies time-based interpolation to missing values. It also examines outliers rather than automatically deleting high-demand observations.

## Workflow

1. **Data loading and quality checks** – load the PJMW workbook, inspect types and summary statistics, identify duplicate timestamps and gaps.
2. **Preprocessing** – resolve duplicate timestamps, reindex to a continuous hourly timeline, and interpolate missing time-series values.
3. **Exploratory data analysis** – study demand over time and compare hourly, weekday/weekend, monthly, seasonal, holiday, and hour-by-season patterns.
4. **Daily aggregation** – resample hourly demand to daily average demand for the daily forecasting experiments.
5. **Feature engineering** – create calendar, weekend, holiday, lag, rolling-statistic, and cyclical time features.
6. **Chronological evaluation** – hold out the final year rather than randomly shuffling time-series observations.
7. **Model comparison** – evaluate ARIMA, SARIMA, XGBoost, Random Forest, LSTM, and GRU.
8. **Forecast generation** – use the selected Random Forest model recursively to generate 30 daily predictions.
9. **Dashboard** – explore the forecast, compare model metrics, inspect demand patterns, and download forecast values.

## Models and evaluation

The supplied `model_comparison.csv` reports the following results on the project's held-out test period:

| Model | MAE (MW) | RMSE (MW) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| ARIMA | 578.80 | 772.96 | -0.0183 | 9.89 |
| SARIMA | 563.59 | 776.10 | -0.0266 | 9.35 |
| XGBoost | 257.52 | 349.97 | 0.7912 | 4.33 |
| **Random Forest** | **251.68** | **333.44** | **0.8105** | **4.29** |
| LSTM | 309.64 | 395.26 | 0.7337 | 5.38 |
| GRU | 292.13 | 372.79 | 0.7631 | 5.09 |

For this experiment, Random Forest has the lowest MAE and RMSE and the highest R² among the six listed models. These are experiment-specific results, not a guarantee of performance on future periods.

**Evaluation caveat:** The dashboard's own note states that ARIMA and SARIMA forecast the full test year in one go, while XGBoost, Random Forest, LSTM, and GRU use the previous day's actual value. The model groups therefore do not use identical forecast procedures, so their scores should be compared with that limitation in mind.

## 30-day forecast

The supplied forecast file is `30_day_forecast.csv` and contains:

- `Date`
- `Predicted_MW`

It contains 30 daily predictions from **3 August 2018 through 1 September 2018**. The mean predicted demand is approximately **5,589.46 MW**.

The notebook generates this forecast recursively: each day's prediction is appended to the history and used when constructing features for the next date. Consequently, errors can accumulate across the multi-day horizon.

## Feature importance

The supplied `random_forest_feature_importance.csv` ranks the features used by the saved feature-importance output. The leading feature is `lag_1` (importance approximately 0.7644), followed by `dayofweek` (approximately 0.0404).

Feature importance indicates how the fitted model used features; it does not establish causation.

## Streamlit dashboard

The application (`app.py`) includes:

- Forecast horizon selector
- Line and bar chart options
- Weekend highlighting
- Optional animated forecast reveal
- Peak, minimum, and average demand metrics
- Weekday and day-over-day forecast insights
- Model-comparison chart and metric selector
- Random Forest feature-importance chart
- CSV download of the selected forecast horizon

## Project structure

Keep the following files together in the application folder:

```text
Hourly-Energy-Consumption-Forecast/
├── app.py
├── Project.ipynb
├── PJMW_MW_Hourly.xlsx
├── 30_day_forecast.csv
├── model_comparison.csv
├── random_forest_feature_importance.csv
├── random_forest_model.pkl       # optional for the dashboard's current code
├── lstm_state_dict.pt            # saved notebook artifact
├── gru_state_dict.pt             # saved notebook artifact
├── requirements.txt
└── README.md
```

The Streamlit app requires `30_day_forecast.csv` and `model_comparison.csv` in the same directory as `app.py`. It can display feature importance from `random_forest_model.pkl` or, if that model file is unavailable, from `random_forest_feature_importance.csv`. A background image (`bg.jpg`, `bg.jpeg`, `bg.png`, or `bg.webp`) is optional.

The app code references `random_forest_model.pkl`; the uploaded project materials include LSTM and GRU state dictionaries, but the Random Forest pickle should also be included if you want the app to load the model directly.

## Installation

Use Python 3.10 or a compatible Python version with the listed packages.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```cmd
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, create one with the packages used by the notebook and dashboard, for example:

```text
numpy
pandas
matplotlib
seaborn
openpyxl
scikit-learn
statsmodels
xgboost
torch
joblib
plotly
streamlit
holidays
```

## Run the dashboard

From the folder containing `app.py` and the required CSV files:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Notebook

Open `Project.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab to review the end-to-end workflow, including preprocessing, EDA, feature engineering, model training, evaluation, and recursive forecast generation.

## Limitations and responsible interpretation

- The delivered 30-day output is **daily average demand**, not an hourly 30-day trajectory.
- The forecast shown in the supplied file is for 3 August–1 September 2018, immediately following the historical data supplied; it is a historical project output, not a live forecast.
- Weather and other external drivers are not included in the described feature set.
- Recursive multi-day forecasting can accumulate prediction errors.
- The evaluation procedures differ between the statistical and machine-learning model groups, as noted above.
- This is an educational forecasting project and should not be treated as an operational grid-dispatch or energy-market system.

## Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Statsmodels · XGBoost · PyTorch · Plotly · Streamlit · Jupyter Notebook

## Author

**Lokesh Chinthala**  
GitHub: [Lokesh-lLoki](https://github.com/Lokesh-lLoki)
