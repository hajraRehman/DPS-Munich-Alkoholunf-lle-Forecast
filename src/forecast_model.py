import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the processed train data (up to 2020)
train = pd.read_csv('data/train_data.csv', parse_dates=['DATE'], index_col='DATE')['WERT']

# Load the future for error calculation
future = pd.read_csv('data/future_data.csv', parse_dates=['DATE'], index_col='DATE')['WERT']

# Fit SARIMA model
# order=(p,d,q) = (1,1,1) — common starting point for non-stationary data
# seasonal_order=(P,D,Q,s) = (1,1,1,12) — captures yearly seasonality (s=12 months)
model = SARIMAX(train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

print(model_fit.summary().tables[1])  # Show key parameters

# Forecast for January 2021 (first month after training period)
forecast_steps = 1
forecast_result = model_fit.get_forecast(steps=forecast_steps)
predicted_value = int(round(forecast_result.predicted_mean.iloc[0]))

print(f"\nPredicted accidents for January 2021: {predicted_value}")

# Actual value from dataset (ground truth)
actual_2021_01 = future.loc['2021-01-01']
print(f"Actual accidents for January 2021: {int(actual_2021_01)}")

# Error
mae = abs(predicted_value - actual_2021_01)
print(f"Absolute Error (MAE): {mae}")

