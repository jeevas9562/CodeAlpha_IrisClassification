import numpy as np

def add_features(df):
    # Avoid division by zero
    df["Km_per_Year"] = df["Driven_kms"] / (df["Car_Age"] + 1)

    return df
