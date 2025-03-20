from typing import (
    Callable,
    Dict,
    Optional,
)

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from matplotlib_plots_utils import (
    add_data_labels,
    add_heatmap_annotations,
    add_scatterplot_average,
    calc_yticks_padding,
    format_axis,
    get_element_coordinates,
    is_legend_empty,
    LEGEND_Y_LOCATION,
)

SEABORN_RC_PARAMS = {
    'font.size': 6,
    'xtick.color': 'gray',
    'xtick.labelsize': 6,
    'ytick.color': 'gray',
    'ytick.labelsize': 6,
    'text.color': 'gray',
    'legend.fontsize': 6,
    'legend.loc': 'upper left',
    'legend.frameon': False,
    'xaxis.labellocation': 'left',
    'yaxis.labellocation': 'top',
    'axes.edgecolor': 'gray',
    'axes.labelcolor': 'gray',
    'axes.titlelocation': 'left',
    'axes.labelsize': 6,
    'axes.titlesize': 12,
    'axes.titlepad': 20,
    'figure.figsize': (6, 3),
    'figure.dpi': 200,
    'figure.titlesize': 12,
}
SEABORN_STYLE = 'white'


def seaborn_plot(
    df: pd.DataFrame,
    function: Callable,
    function_kws: Dict,
    title: Optional[str] = None,
    xlabel_kws: Optional[Dict] = None,
    xticks_kws: Optional[Dict] = None,
    xlim_kws: Optional[Dict] = None,
    ylabel_kws: Optional[Dict] = None,
    yticks_kws: Optional[Dict] = None,
    ylim_kws: Optional[Dict] = None,
    pad_yticks: bool = False,
    despine_kws: Optional[Dict] = None,
    is_subplot: bool = False,
    legend_kws: Optional[Dict] = None,
    data_labels_kws: Optional[Dict] = None,
    plot_average: bool = False,
    align_with: str = 'yaxis_label',
    coordinate_system: str = 'axes',
    ax: Optional[Axes] = None,
):
    ax = function(data=df, ax=ax, **function_kws)
    format_axis(ax, xlabel_kws, xticks_kws, xlim_kws, ylabel_kws, yticks_kws, ylim_kws)
    if despine_kws:
        sns.despine(**despine_kws)
    ax.get_figure().canvas.draw()
    # Align yticks
    if pad_yticks:
        ax.get_yaxis().set_tick_params(pad=calc_yticks_padding(ax))
    if is_subplot:
        if not is_legend_empty(ax.get_legend_handles_labels()):
            ax.get_legend().remove()
    else:
        # Align legend
        x, y = get_element_coordinates(ax, align_with, coordinate_system)
        if not is_legend_empty(ax.get_legend_handles_labels()):
            handles, labels = ax.get_legend_handles_labels()
            if plot_average and function == sns.scatterplot:
                handles, labels = add_scatterplot_average(
                    df[function_kws['x']].mean(), df[function_kws['y']].mean(),
                    handles, labels,
                )
            ax.legend(handles, labels, ncol=len(labels), bbox_to_anchor=(x, LEGEND_Y_LOCATION), **legend_kws)
        if function == sns.heatmap:
            add_heatmap_annotations(
                ax,
                x,
                sns.color_palette(function_kws['cmap'])[1],
                sns.color_palette(function_kws['cmap'])[-1],
            )
        # Align title
        ax.set_title(title, x=x)
    if data_labels_kws:
        add_data_labels(ax, function_kws, data_labels_kws)