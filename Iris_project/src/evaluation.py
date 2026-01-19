from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_confusion(y_test, y_pred, labels, ax=None, fig=None, streamlit=True):
   
    import matplotlib.pyplot as plt
    import seaborn as sns

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7,6))
    
    cm = confusion_matrix(y_test, y_pred)
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket",
        linewidths=0.5,
        annot_kws={"color": "white", "size": 11},
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax
    )

    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    ax.set_xlabel("Predicted", color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", color="white", fontsize=13, fontweight="bold")

    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors="white", labelsize=10)
    cbar.outline.set_edgecolor("white")

    if streamlit:
        import streamlit as st
        st.pyplot(fig, use_container_width=False)
    else:
        plt.show()
def plot_pca(X, y, streamlit=True):
    """
    Plots a 2D PCA feature projection.
    
    X : feature dataframe or array
    y : target labels
    streamlit : if True, displays using Streamlit
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.85
    )

    ax.set_xlabel("PCA Component 1", color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("PCA Component 2", color="white", fontsize=12, fontweight="bold")

    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white", labelsize=10)

    if streamlit:
        import streamlit as st
        st.pyplot(fig, use_container_width=False)
    else:
        plt.show()
