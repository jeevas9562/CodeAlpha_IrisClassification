def covid_impact(df):
    df['Covid_Period'] = df['Date'].dt.year >= 2020
    return df.groupby('Covid_Period')['Unemployment Rate'].mean()


def region_analysis(df):
    if 'Region' in df.columns:
        return df.groupby('Region')['Unemployment Rate'].mean()
    return None
