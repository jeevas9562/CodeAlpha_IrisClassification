import numpy as np
from sklearn.linear_model import LinearRegression

def predict_next_year_unemployment(df, final_year=None):
    """
    Predict unemployment rate using Linear Regression.
    """

    # Convert to yearly average
    yearly_data = df.groupby(df['Date'].dt.year)['Unemployment Rate'].mean().reset_index()
    yearly_data.columns = ['Year', 'Unemployment Rate']

    X = yearly_data['Year'].values.reshape(-1, 1)
    y = yearly_data['Unemployment Rate'].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    last_year = int(X.max())
    if final_year is None:
        final_year = last_year + 3

    future_years = np.arange(X.min(), final_year + 1).reshape(-1, 1)
    future_pred = model.predict(future_years)

    next_year = future_years[-1][0]
    predicted = future_pred[-1]

    return next_year, predicted, X, y, future_years, future_pred
