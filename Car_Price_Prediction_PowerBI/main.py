import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import preprocess_data
from src.feature_engineering import add_features
from src.train_model import train_random_forest
from src.evaluate_model import evaluate_model

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/car data.csv")

# -------------------------------
# Preprocessing
# -------------------------------
df = preprocess_data(df)
df = add_features(df)

# -------------------------------
# Feature Selection
# -------------------------------
X = df[
    [
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age",
        "Km_per_Year"
    ]
]

y = df["Selling_Price"]

# -------------------------------
# Train Model
# -------------------------------
model, X_test, y_test = train_random_forest(X, y)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
metrics = evaluate_model(y_test, y_pred)

print("Model Performance:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -------------------------------
# Prepare Power BI Output
# -------------------------------
results = X_test.copy()
results["ActualPrice"] = y_test.values
results["PredictedPrice"] = y_pred
results["Residual"] = results["ActualPrice"] - results["PredictedPrice"]

def price_range(price):
    if price < 3:
        return "Budget"
    elif price < 6:
        return "Mid"
    else:
        return "Premium"

results["PriceRange"] = results["ActualPrice"].apply(price_range)

results.to_csv("outputs/car_price_predictions.csv", index=False)

# -------------------------------
# Feature Importance Plot
# -------------------------------
importance = model.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance - Car Price Prediction")
plt.tight_layout()
plt.show()

from src.visualization import (
    actual_vs_predicted,
    residual_plot,
    feature_importance_plot
)
import os

# -------------------------------
# Visualization Export
# -------------------------------
visual_path = "outputs/visuals"
os.makedirs(visual_path, exist_ok=True)

actual_vs_predicted(y_test, y_pred, visual_path)
residual_plot(y_test, y_pred, visual_path)
feature_importance_plot(model, X.columns, visual_path)
