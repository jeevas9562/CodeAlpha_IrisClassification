# src/pie_chart_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

class PieChartVisualizer:
    def __init__(self, data, title="", top_n=None, combine_others=True,
                 colors=None, dark_mode=False, transparent=False):
        """
        Args:
            data (pd.Series): values to plot, index = labels
            title (str): chart title
            top_n (int): show only top N items (rest combined into 'Others')
            combine_others (bool): whether to combine smaller items into 'Others'
            colors (list): optional list of colors for slices
            dark_mode (bool): use dark background
            transparent (bool): make figure background transparent
        """
        self.data = data
        self.title = title
        self.top_n = top_n
        self.combine_others = combine_others
        self.colors = colors
        self.dark_mode = dark_mode
        self.transparent = transparent

    def prepare_data(self):
        data = self.data.copy()
        if self.top_n and self.combine_others:
            top_data = data.sort_values(ascending=False).head(self.top_n)
            others_sum = data.sum() - top_data.sum()
            if others_sum > 0:
                top_data["Others"] = others_sum
            data = top_data
        return data

    def get_color_palette(self, n_colors):
        """Return a professional color palette."""
        if self.colors:
            return self.colors  # use user-defined colors
        # Use Matplotlib qualitative palettes
        if self.dark_mode:
            # Dark-friendly palette
            colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        else:
            # Light-friendly palette
            colors = list(mcolors.TABLEAU_COLORS.values())
        # Repeat if needed
        if n_colors > len(colors):
            import itertools
            colors = list(itertools.islice(itertools.cycle(colors), n_colors))
        return colors[:n_colors]

    def plot(self, use_streamlit=True, figsize=(6,6)):
        data = self.prepare_data()
        n_colors = len(data)
        colors = self.get_color_palette(n_colors)

        # Set dark mode style
        if self.dark_mode:
            plt.style.use('dark_background')
            text_color = 'white'
            legend_facecolor = 'black'
            legend_edgecolor = 'white'
        else:
            plt.style.use('default')
            text_color = 'black'
            legend_facecolor = 'white'
            legend_edgecolor = 'black'

        fig, ax = plt.subplots(figsize=figsize)

        # Plot pie chart with labels and percentages
        wedges, texts, autotexts = ax.pie(
            data,
            labels=data.index,  # Show region names directly on slices
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 10, 'color': text_color},
            labeldistance=1.05
        )

        # Improve autotext visibility
        for autotext in autotexts:
            autotext.set_color(text_color)
            autotext.set_fontsize(10)

        # Set title
        ax.set_title(self.title, color=text_color)

        # Add legend
        ax.legend(
            wedges,
            data.index,
            title="Region",
            loc="best",
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            labelcolor=text_color
        )

        # Transparent background if requested
        if self.transparent:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

        # Display chart
        if use_streamlit:
            st.pyplot(fig, use_container_width=True)
        else:
            plt.show()

        return fig
