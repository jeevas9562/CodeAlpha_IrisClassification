import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from src.evaluation import evaluate, plot_confusion, plot_pca
from src.data_preprocessing import load_and_preprocess
from src.visualization import pair_plot, box_plot
from src.prediction import predict_species


# ---------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    layout="wide"
)

# ---------------------------
# CUSTOM DARK UI STYLING
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
body {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #7C83FD;
}
.glass-card {
    background: rgba(30, 30, 47, 0.65);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    text-align: center;
}
.metric-title {
    font-size: 18px;
    color: #B8B8FF;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.title("🌸 Iris Flower Classification Dashboard")
st.markdown("**Modern Machine Learning Dashboard using Streamlit**")

# ---------------------------
# LOAD DATA
# ---------------------------
# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    # Original data for display
    df = pd.read_csv("data/Iris.csv")
    df.drop("Id", axis=1, inplace=True)
    return df

df = load_data()  # for Dataset Overview & visualizations

# Preprocess separately for modeling
X, y, le = load_and_preprocess("data/Iris.csv")


# ---------------------------
# SPECIES IMAGE FOLDERS
# ---------------------------
species_image_dirs = {
    "Iris-setosa": "images/Iris-setosa",
    "Iris-versicolor": "images/Iris-versicolor",
    "Iris-virginica": "images/Iris-virginica"
}

# ---------------------------
# PREPROCESSING
# ---------------------------
X, y, le = load_and_preprocess("data/Iris.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MODEL TRAINING
# ---------------------------
from src.models import train_logistic, train_knn

lr, lr_pred, lr_acc = train_logistic(X_train, y_train, X_test, y_test)
knn, knn_pred, knn_acc = train_knn(X_train, y_train, X_test, y_test)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown("## 🔍 Dashboard ")
st.sidebar.markdown("---")

option = st.sidebar.radio(
    "Choose a section",
    [
        "Dataset Overview",
        "Model Performance",
        "Visualizations",
        "Predict Species"
    ]
)

# ---------------------------
# DATASET OVERVIEW
# ---------------------------
st.markdown("""
<style>
.dataset-table div {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


if option == "Dataset Overview":
    st.markdown('<div class="dataset-table">', unsafe_allow_html=True)

    # Show first 5 rows of each species for a balanced overview
    st.dataframe(pd.concat([
        df[df['Species'] == 'Iris-setosa'].head(5),
        df[df['Species'] == 'Iris-versicolor'].head(5),
        df[df['Species'] == 'Iris-virginica'].head(5)
    ]), height=400, use_container_width=True)

    # Show summary statistics (numerical columns)
    st.dataframe(df.describe(), height=350, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# MODEL PERFORMANCE
# ---------------------------
elif option == "Model Performance":
    st.markdown("<h2 style='font-weight:bold'>📈 Model Accuracy Comparison</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Logistic Regression Accuracy</div>
            <div class="metric-value">{lr_acc:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">KNN Accuracy</div>
            <div class="metric-value">{knn_acc:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-weight:bold'>Accuracy Bar Chart</h2>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6,2))
    ax.bar(["Logistic Regression", "KNN"], [lr_acc, knn_acc], width=0.4)  # narrower bars
    ax.set_ylim(0.9, 1.0)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    st.pyplot(fig)

# ---------------------------
# VISUALIZATIONS
# ---------------------------
elif option == "Visualizations":
    st.subheader("📊 Feature Visualizations")

    # PCA
    st.markdown("### PCA – 2D Feature Projection")

    # Call the PCA function
    plot_pca(X, y, streamlit=True)

    # Confusion Matrix
    # Use the predictions already computed
    st.markdown("### Confusion Matrix (Logistic Regression)")
    plot_confusion(y_test, lr_pred, labels=le.classes_, streamlit=True)

   
      # Pair Plot (full width)
    st.markdown("### Pair Plot")
    pair_plot(df, height=5, aspect=1)  # taller

    # Box Plots (2 columns)
    st.markdown("### Box Plots")
    box_plot(df, figsize=(5,3))  # smaller plots in 2 columns
# ---------------------------
# PREDICTION
# ---------------------------
elif option == "Predict Species":
    predict_species(lr, le, species_image_dirs)
