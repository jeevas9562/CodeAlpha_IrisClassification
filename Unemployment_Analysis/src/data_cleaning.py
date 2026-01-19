import pandas as pd

def clean_data(df):
    df.columns = df.columns.str.strip()

    # Date handling
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'Month' in df.columns:
        date_col = 'Month'
    else:
        raise KeyError("No Date or Month column found")

    df[date_col] = pd.to_datetime(df[date_col])
    df.rename(columns={date_col: 'Date'}, inplace=True)
    df = df.sort_values(by='Date')

    # Unemployment column standardization
    for col in df.columns:
        if 'Unemployment' in col:
            df.rename(columns={col: 'Unemployment Rate'}, inplace=True)

    return df
