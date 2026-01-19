import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.analysis import covid_impact, region_analysis
from src.prediction import predict_next_year_unemployment
from src.visualization import (
    unemployment_trend,
    covid_bar_plot,
    region_plot,
    covid_pie_chart,
    region_pie_chart
)

st.set_page_config(page_title="Unemployment Analysis", layout="wide")

st.title("📊 Unemployment Analysis Dashboard")
st.markdown("Analyze unemployment trends and Covid-19 impact")

# Load data
df = load_data("data/unemployment.csv")
df = clean_data(df)

st.sidebar.subheader("📅 Filters")

year = st.sidebar.selectbox(
    "Select Year",
    sorted(df['Date'].dt.year.unique())
)

filtered_df = df[df['Date'].dt.year == year]

# ================= SIDEBAR =================
st.sidebar.title("📌 Dashboard Menu")

section = st.sidebar.radio(
    "Choose Analysis Section",
    (
        "📁 Dataset Overview",
        "📉 Unemployment Trend",
        "🦠 Covid-19 Impact",
        "🌍 Region-wise Analysis",
        "📊 Predicted Unemployment" 
    )
)

# ================= MAIN CONTENT =================

# 1️⃣ Dataset Overview
if section == "📁 Dataset Overview":
    st.header("📁 Dataset Overview")
    st.subheader("Preview of Dataset")
    st.dataframe(df.head(20), height=400)  # first 20 rows + scrollable

    st.subheader("Statistical Summary")
    st.write(df.describe())

# 2️⃣ Unemployment Trend
elif section == "📉 Unemployment Trend":
    st.header("📉 Unemployment Trend Over Time")
    fig = unemployment_trend(filtered_df)
    fig.set_size_inches(7,2.5)
    st.pyplot(fig)

# 3️⃣ Covid Impact
elif section == "🦠 Covid-19 Impact":
    st.header("🦠 Covid-19 Impact on Unemployment")

    covid_data = covid_impact(df)

    # Original bar chart
    fig_bar = covid_bar_plot(covid_data)
    st.subheader("Covid-19 Bar Chart")
    st.pyplot(fig_bar, use_container_width=True)

    # Stylish pie chart
    fig_pie = covid_pie_chart(covid_data, dark_mode=True)
    st.subheader("Covid-19 Pie Chart")
    st.pyplot(fig_pie, use_container_width=True)

    st.success(
        "Unemployment rates increased significantly during the Covid-19 period."
    )

# 4️⃣ Region-wise Analysis
elif section == "🌍 Region-wise Analysis":
    st.header("🌍 Region-wise Unemployment Analysis")

    region_data = region_analysis(filtered_df)

    if region_data is not None and not region_data.empty:
        # Original bar chart
        fig_bar = region_plot(region_data)
        st.subheader("Region-wise Bar Chart")
        st.pyplot(fig_bar)

        # Stylish pie chart
        fig_pie = region_pie_chart(region_data, dark_mode=True, top_n=5)
        st.subheader("Region-wise Pie Chart")
        st.pyplot(fig_pie, use_container_width=True)
    else:
        st.warning("Region column not found or empty in dataset.")

# 5️⃣ Predicted Unemployment
elif section == "📊 Predicted Unemployment":
    st.header("📊 Predicted Unemployment Rate")

    from src.prediction import predict_next_year_unemployment
    from src.visualization import unemployment_trend  # Updated function

    # 🔹 Use prediction.py directly
    next_year, predicted, X_hist, y_hist, future_years, future_pred = predict_next_year_unemployment(df, final_year=2028)

    # Show predicted value
    st.info(f"Predicted average unemployment rate for **{int(next_year)}**: **{predicted:.2f}%**")

    # 🔹 Plot historical + predicted trend with dark-mode styling
    fig = unemployment_trend(
        df,
        future_years=future_years.flatten(),
        future_pred=future_pred,
        dark_mode=True
    )

    # Display in Streamlit
    st.pyplot(fig, use_container_width=True)

