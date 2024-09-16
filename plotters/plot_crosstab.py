
import matplotlib.ticker as mtick
from util.plot import plot_config, add_bar_labels


def plot_crosstab(crosst, ax, **kwds):

    # Load configuration
    # plot_config()

    # Plot stacked bars
    crosst.plot.bar(stacked=True, ax=ax, **kwds)

    # Add data labels
    add_bar_labels(ax)

    # Configure axis
    ax.legend(
        title=crosst.columns.name,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        handlelength=0.8,
        alignment="left",
    )
    ax.set(xlabel=None, title=crosst.index.name)

    # Format yticklabels
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # Return result
    return ax
