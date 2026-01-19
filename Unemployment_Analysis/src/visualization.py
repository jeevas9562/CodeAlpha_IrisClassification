import matplotlib.pyplot as plt
import seaborn as sns

# Set global dark style for seaborn/matplotlib
sns.set_style("darkgrid")
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': 'white',
    'axes.facecolor': '#2c2c2c',  # dark gray background
    'figure.facecolor': '#2c2c2c',  # figure background
})

# ------------------------
# Unemployment trend line
# ------------------------
import matplotlib.pyplot as plt

def unemployment_trend(df, future_years=None, future_pred=None, dark_mode=False):
    fig, ax = plt.subplots(figsize=(7, 2.5))

    if dark_mode:
        fig.patch.set_alpha(0)
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Historical data
    ax.plot(
        df['Date'],
        df['Unemployment Rate'],
        marker='o',
        label="Historical",
        color="#66b3ff"
    )

    # Prediction data
    if future_years is not None and future_pred is not None:
        ax.plot(
            future_years,
            future_pred,
            linestyle='--',
            marker='x',
            label="Predicted",
            color="#ff9999"
        )

    ax.set_title("Unemployment Rate Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()

    plt.tight_layout()
    return fig


# ------------------------
# Covid bar plot
# ------------------------
def covid_bar_plot(data):
    fig, ax = plt.subplots(figsize=(5,2.5))
    palette = sns.color_palette("Set2", 2)
    sns.barplot(x=data.index, y=data.values, ax=ax, palette=palette)
    ax.set_title("Before vs After Covid-19", fontsize=13)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_xticklabels(["Before Covid", "After Covid"], rotation=0, color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

# ------------------------
# Covid pie chart (donut)
# ------------------------

def covid_pie_chart(data, dark_mode=True, transparent=False):
    """
    Stylish donut-style Covid-19 pie chart with dark/light mode
    """
    fig, ax = plt.subplots(figsize=(3,3))
    colors = sns.color_palette("Set2", len(data))

    wedges, texts, autotexts = ax.pie(
        data,
        labels=["Before Covid", "After Covid"],
        autopct='%1.1f%%',
        pctdistance=0.85,
        startangle=140,
        explode=[0.05]*len(data),
        shadow=True,
        wedgeprops={'edgecolor':'white', 'linewidth':1.5},
        colors=colors
    )

    # Donut effect
    centre_color = '#2c2c2c' if dark_mode else 'white'
    centre_circle = plt.Circle((0,0),0.70,fc=centre_color, alpha=0.85)
    fig.gca().add_artist(centre_circle)

    # Style text
    text_color = 'white' if dark_mode else 'black'
    for t in texts + autotexts:
        t.set_color(text_color)
        t.set_fontsize(12)
        t.set_fontweight('bold')

    ax.set_title("Covid-19 Impact on Unemployment", color=text_color, fontsize=14)

    # Transparent background
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    # Dark mode figure background
    if dark_mode:
        ax.set_facecolor('#2c2c2c')
        fig.patch.set_facecolor('#2c2c2c')

    return fig

# ------------------------
# Region pie chart (donut) with top-N and highlight
# ------------------------
def region_pie_chart(region_data, dark_mode=True, transparent=False, top_n=5):
    """
    Stylish donut-style region-wise pie chart with professional colors and dark/light mode
    """
    # Keep top N regions + combine others
    data = region_data.sort_values(ascending=False)
    if top_n and len(data) > top_n:
        top_data = data.head(top_n)
        others = data.sum() - top_data.sum()
        top_data["Others"] = others
        data = top_data

    n_colors = len(data)
    colors = sns.color_palette("tab20", n_colors)

    # Highlight largest slice
    explode = [0.1 if i==0 else 0.05 for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(4,4))

    wedges, texts, autotexts = ax.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        pctdistance=0.85,
        startangle=140,
        explode=explode,
        shadow=True,
        wedgeprops={'edgecolor':'white', 'linewidth':1.5},
        colors=colors
    )

    # Donut center with subtle transparency
    centre_color = '#2c2c2c' if dark_mode else 'white'
    centre_circle = plt.Circle((0,0),0.70,fc=centre_color, alpha=0.85)
    fig.gca().add_artist(centre_circle)

    # Style text
    text_color = 'white' if dark_mode else 'black'
    for t in texts + autotexts:
        t.set_color(text_color)
        t.set_fontsize(12)
        t.set_fontweight('bold')

    ax.set_title("Proportion of Unemployment by Region", color=text_color, fontsize=14)

    # Transparent background
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    # Dark mode figure background
    if dark_mode:
        ax.set_facecolor('#2c2c2c')
        fig.patch.set_facecolor('#2c2c2c')

    return fig


# ------------------------
# Region bar plot
# ------------------------
def region_plot(data):
    fig, ax = plt.subplots(figsize=(6,3))
    colors = sns.color_palette("tab20", len(data))
    sns.barplot(x=data.index, y=data.values, ax=ax, palette=colors)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=10, color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig
