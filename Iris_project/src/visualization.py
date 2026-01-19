import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def pair_plot(df, streamlit=True, height=6, aspect=1):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # height: height of each subplot, aspect: width/height ratio
    g = sns.pairplot(df, hue="Species", height=height, aspect=aspect)
    
    # Reduce spacing between plots
    g.fig.subplots_adjust(wspace=0.2, hspace=0.2)
    
    if streamlit:
        st.pyplot(g.fig)
    else:
        plt.show()

def box_plot(df, streamlit=True, figsize=(5,3)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    features = df.columns[:-1]  # all features except Species

    # Split features into 2 columns
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x="Species", y=feature, data=df, ax=ax)
        plt.tight_layout()

        # Alternate columns
        if i % 2 == 0:
            if streamlit:
                with col1:
                    st.pyplot(fig)
            else:
                plt.show()
        else:
            if streamlit:
                with col2:
                    st.pyplot(fig)
            else:
                plt.show()
