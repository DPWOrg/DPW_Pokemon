from typing import (
    Dict,
    Optional,
)

import pandas as pd
import seaborn as sns

from Jason.My_utils.seaborn_plots_utils.matplotlib_plots_utils import (
    add_data_labels,
    calc_yticks_padding,
    format_axis,
    get_element_coordinates,
    is_legend_empty,
    LEGEND_Y_LOCATION,
)


def pandas_plot(
    df: pd.DataFrame,
    plot_kws: Dict,
    title: Optional[str] = None,
    xlabel_kws: Optional[Dict] = None,
    xticklabels_kws: Optional[Dict] = None,
    xlim_kws: Optional[Dict] = None,
    ylabel_kws: Optional[Dict] = None,
    yticklabels_kws: Optional[Dict] = None,
    ylim_kws: Optional[Dict] = None,
    pad_yticks: bool = False,
    invert_yaxis: bool = False,
    despine_kws: Optional[Dict] = None,
    legend_kws: Optional[Dict] = None,
    data_labels_kws: Optional[Dict] = None,
    align_with: str = 'yaxis_label',
    coordinate_system: str = 'axes',
):
    ax = df.plot(**plot_kws)
    if invert_yaxis:
        ax.invert_yaxis()
    format_axis(ax, xlabel_kws, xticklabels_kws, xlim_kws, ylabel_kws, yticklabels_kws, ylim_kws)
    if despine_kws:
        sns.despine(**despine_kws)
    ax.get_figure().canvas.draw()
    # Align yticks
    if pad_yticks:
        ax.get_yaxis().set_tick_params(pad=calc_yticks_padding(ax))
    # Align legend and title
    x, y = get_element_coordinates(ax, align_with, coordinate_system)
    if not is_legend_empty(ax.get_legend_handles_labels()):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=len(labels), bbox_to_anchor=(x, LEGEND_Y_LOCATION), **legend_kws)
    if title:
        ax.set_title(title, x=x)
    if data_labels_kws:
        add_data_labels(ax, plot_kws, data_labels_kws)