import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    # Create car age
    current_year = datetime.now().year
    df["Car_Age"] = current_year - df["Year"]

    # Drop unused columns
    df.drop(["Year", "Car_Name"], axis=1, inplace=True)

    # Encode categorical columns
    categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df
