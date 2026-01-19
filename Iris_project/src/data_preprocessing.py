import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.drop("Id", axis=1, inplace=True)

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])

    X = df.drop("Species", axis=1)
    y = df["Species"]

    return X, y, le
