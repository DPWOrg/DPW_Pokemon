import matplotlib.pyplot as plt
import pandas as pd
import squarify


def treemap(
    series: pd.core.series.Series,
    title: str,
    **squarify_args,
):
    ax = squarify.plot(series, **squarify_args)
    ax.set_title(title)
    plt.axis('off')