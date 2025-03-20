from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import IdentityTransform
from matplotlib.axes import Axes

DEFAULT_SUP_LEGEND_Y_LOCATION = 0.94
DEFAULT_SUPTITLE_Y_LOCATION = 1
DEFAULT_SUPTITLE_HA = 'left'

LEGEND_Y_LOCATION = 1.1
HEATMAP_ANNOTATION_Y_LOCATION = 1.05


# def calc_yticks_padding(ax: Axes) -> float:
#     return max(T.label.get_window_extent().width for T in ax.get_yaxis().majorTicks) * 0.5



def calc_yticks_padding(ax: Axes) -> float:
    # 确保绘图已渲染以获取正确标签尺寸
    fig = ax.figure
    fig.canvas.draw()  # 关键：强制渲染

    # 直接获取 y 轴标签的 Text 对象
    ytick_labels = ax.get_yticklabels()

    # 计算最大标签宽度
    max_width = max(
        label.get_window_extent().width
        for label in ytick_labels
    )

    return max_width * 0.5  # 按需调整缩放比例


def is_legend_empty(legend_handles_labels: Tuple[List, List]) -> bool:
    return not any(legend_handles_labels)


def get_element_coordinates(ax: Axes, align_with: str, coordinate_system: str) -> Tuple[float, float]:
    if align_with == 'ytick_labels':
        bbox = ax.get_yticklabels()[-1].get_window_extent()
    elif align_with == 'xaxis_label':
        bbox = ax.xaxis.get_label().get_window_extent()
    elif align_with == 'xtick_labels':
        bbox = ax.get_xticklabels()[0].get_window_extent()
    else:
        bbox = ax.yaxis.get_label().get_window_extent()
    if coordinate_system == 'axes':
        return ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
    else:
        return ax.get_figure().transFigure.inverted().transform([bbox.x0, bbox.y0])


def format_axis(
    ax: Axes,
    xlabel_kws: Optional[Dict] = None,
    xticks_kws: Optional[Dict] = None,
    xlim_kws: Optional[Dict] = None,
    ylabel_kws: Optional[Dict] = None,
    yticks_kws: Optional[Dict] = None,
    ylim_kws: Optional[Dict] = None,
):
    if xlabel_kws:
        ax.set_xlabel(**xlabel_kws)
    if xticks_kws:
        ticks = xticks_kws.get('ticks', ax.get_xticks())
        xticks_kws.pop('ticks', None)
        labels = xticks_kws.get('labels', ax.get_xticklabels())
        xticks_kws.pop('labels', None)
        ax.set_xticks(ticks, labels=labels, **xticks_kws)
    if xlim_kws:
        ax.set_xlim(**xlim_kws)
    if ylabel_kws:
        ax.set_ylabel(**ylabel_kws)
    if yticks_kws:
        ticks = yticks_kws.get('ticks', ax.get_yticks())
        yticks_kws.pop('ticks', None)
        labels = yticks_kws.get('labels', ax.get_yticklabels())
        yticks_kws.pop('labels', None)
        ax.set_yticks(ticks, labels=labels, **yticks_kws)
    if ylim_kws:
        ax.set_ylim(**ylim_kws)


def add_heatmap_annotations(ax: Axes, x_location: float, low_color: str, high_color: str):
    ax.annotate(
        'LOW',
        xy=(x_location, HEATMAP_ANNOTATION_Y_LOCATION),
        xytext=(x_location, HEATMAP_ANNOTATION_Y_LOCATION),
        color=low_color,
        size=6,
        xycoords='axes fraction',
    )
    ax.annotate(
        'HIGH',
        xy=(x_location + 0.1, HEATMAP_ANNOTATION_Y_LOCATION),
        xytext=(x_location + 0.1, HEATMAP_ANNOTATION_Y_LOCATION),
        color=high_color,
        size=6,
        xycoords='axes fraction',
    )


def add_scatterplot_average(x: float, y: float, handles: List, labels: List) -> Tuple[List, List]:
    plt.scatter(x=x, y=y, color="gray")
    plt.axvline(x=x, color="gray", linestyle='dotted')
    plt.axhline(y=y, color="gray", linestyle='dotted')
    handles.append(plt.Line2D([], [], color="gray", marker="o", linewidth=0))
    labels.append('Average')
    return handles, labels


def add_data_labels(ax: Axes, function_kws: Dict, data_labels_kws: Dict):
    if 'hue' in function_kws.keys() or ('kind' in function_kws.keys() and function_kws['stacked']):
        for container in ax.containers:
            ax.bar_label(container, **data_labels_kws)
    else:
        ax.bar_label(ax.containers[0], **data_labels_kws)


def plot_simple_text_percentage(percentage: int, second_line_text: str, third_line_text: str):
    fig = plt.figure(figsize=(1, 1))
    fig.text(25, 200, f'{percentage}%', color='forestgreen', fontsize=24, transform=IdentityTransform())
    fig.text(25, 150, second_line_text, color='forestgreen', fontsize=12, transform=IdentityTransform())
    fig.text(25, 100, third_line_text, color='lightgrey', fontsize=12, transform=IdentityTransform())
    plt.axis('off')