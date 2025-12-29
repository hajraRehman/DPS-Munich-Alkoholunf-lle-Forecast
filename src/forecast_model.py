import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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


rmse = np.sqrt(mean_squared_error([actual_2021_01], [predicted_value]))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot forecast vs actual for first few years (visual validation)
future_forecast = model_fit.get_forecast(steps=len(future))
forecast_df = future_forecast.predicted_mean
confidence = future_forecast.conf_int()

plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Training Data (2000–2020)', color='blue')
plt.plot(future.index, future, label='Actual (2021–2025)', color='green')
plt.plot(forecast_df.index, forecast_df, label='Forecast (2021–2025)', color='red', linestyle='--')
plt.fill_between(confidence.index, confidence.iloc[:, 0], confidence.iloc[:, 1], color='pink', alpha=0.3)
plt.axvline(pd.to_datetime('2021-01-01'), color='gray', linestyle='--', label='Train/Forecast Split')
plt.title('SARIMA Forecast vs Actual Alcohol-Related Accidents')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sarima_forecast_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nForecast plot saved as sarima_forecast_vs_actual.png")
