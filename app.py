from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)
# Train the model ONCE when the app starts
print("Loading training data and fitting SARIMA model...")

# Load the clean training data (up to 2020)
train = pd.read_csv('train_data.csv', parse_dates=['DATE'], index_col='DATE')['WERT']

# Same winning parameters that gave us prediction=21 and MAE=5
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)
print("SARIMA model trained and ready for predictions!")


# Prediction function
def predict_accidents(year: int, month: int) -> int:
    # Training ends Dec 2020
    # For any date after Dec 2020, calculate months ahead from Jan 2021 as step 1
    base_year = 2021
    base_month = 1
    months_ahead = (year - base_year) * 12 + (month - base_month) + 1  # +1 for the first forecast step

    if months_ahead <= 0:
        return 0  # Invalid or in training period

    forecast = model_fit.get_forecast(steps=months_ahead)
    predicted_mean = forecast.predicted_mean.iloc[-1]
    
    return max(int(round(predicted_mean)), 0)

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        year = int(data['year'])
        month = int(data['month'])
        
        # Basic validation
        if not (2000 <= year <= 2100) or not (1 <= month <= 12):
            return jsonify({"error": "Invalid year or month"}), 400
        
        prediction = predict_accidents(year, month)
        
        return jsonify({
            "year": year,
            "month": month,
            "prediction": prediction
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run local server (for testing)
if __name__ == '__main__':
    app.run(debug=True)