import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PLOT_FOLDER = "app/static/plots"
os.makedirs(PLOT_FOLDER, exist_ok=True)


def generate_plot(df: pd.DataFrame, chart_type: str, columns: list):
    """
    Generate chart image based on chart type + columns
    Returns image URL
    """

    filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(PLOT_FOLDER, filename)

    plt.figure(figsize=(10, 6))

    try:
        # =====================================================
        # 1 COLUMN CHARTS
        # =====================================================
        if chart_type == "histogram":
            df[columns[0]].dropna().plot(kind="hist", bins=20)
            plt.xlabel(columns[0])

        elif chart_type == "boxplot":
            sns.boxplot(y=df[columns[0]])
            plt.ylabel(columns[0])

        elif chart_type == "bar":
            df[columns[0]].value_counts().plot(kind="bar")
            plt.xlabel(columns[0])
            plt.ylabel("Count")

        elif chart_type == "pie":
            df[columns[0]].value_counts().plot(
                kind="pie",
                autopct="%1.1f%%"
            )
            plt.ylabel("")

        elif chart_type == "line":
            df[columns[0]].plot(kind="line")
            plt.ylabel(columns[0])

        # =====================================================
        # 2 COLUMN CHARTS
        # =====================================================
        elif chart_type == "scatter":
            sns.scatterplot(
                x=df[columns[0]],
                y=df[columns[1]]
            )

        elif chart_type == "line_2d":
            sns.lineplot(
                x=df[columns[0]],
                y=df[columns[1]]
            )

        elif chart_type == "area":
            df.plot(
                x=columns[0],
                y=columns[1],
                kind="area"
            )

        elif chart_type == "stacked_bar":
            pd.crosstab(
                df[columns[0]],
                df[columns[1]]
            ).plot(kind="bar", stacked=True)

        elif chart_type == "grouped_bar":
            pd.crosstab(
                df[columns[0]],
                df[columns[1]]
            ).plot(kind="bar")

        elif chart_type == "heatmap":
            cross = pd.crosstab(df[columns[0]], df[columns[1]])
            sns.heatmap(cross, annot=True, fmt="d")

        elif chart_type == "boxplot_2d":
            sns.boxplot(
                x=df[columns[0]],
                y=df[columns[1]]
            )

        # =====================================================
        # 3 COLUMN CHARTS
        # =====================================================
        elif chart_type == "grouped_boxplot":
            sns.boxplot(
                x=df[columns[0]],
                y=df[columns[2]],
                hue=df[columns[1]]
            )

        elif chart_type == "multi_line":
            pivot = df.pivot_table(
                index=columns[0],
                columns=columns[1],
                values=columns[2],
                aggfunc="mean"
            )
            pivot.plot()

        elif chart_type == "stacked_area":
            pivot = df.pivot_table(
                index=columns[0],
                columns=columns[1],
                values=columns[2],
                aggfunc="mean"
            )
            pivot.plot(kind="area", stacked=True)

        elif chart_type == "colored_scatter":
            sns.scatterplot(
                x=df[columns[0]],
                y=df[columns[1]],
                hue=df[columns[2]]
            )

        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # =====================================================
        # COMMON FORMATTING
        # =====================================================
        plt.title(f"{chart_type}: {' vs '.join(columns)}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return {
            "chart": chart_type,
            "columns": columns,
            "image_url": f"/static/plots/{filename}"
        }

    except Exception as e:
        plt.close()
        raise ValueError(f"Failed to generate {chart_type}: {str(e)}")