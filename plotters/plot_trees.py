import pandas as pd
import numpy as np
from util.plot import plot_config


def plot_feature_importance(feature_importance, feature_labels, data_dict, ax, yerr=None, **kwds):

    # Load configuration
    # plot_config()

    # Construct plot
    feature_importances = pd.Series(feature_importance, index=[feature_labels[i] for i in range(len(feature_importance))]).sort_values(ascending=False)
    if type(yerr) is np.ndarray:
        feature_importances.plot.bar(yerr=yerr, ax=ax, **kwds)
    else:
        feature_importances.plot.bar(ax=ax, **kwds)

    # Label bars
    ax.bar_label(ax.containers[0], labels=[f"{fi:.1%}" for fi in feature_importances])

    # Format y axis and labels
    ax.yaxis.set_major_formatter('{x:.0%}')
    ax.set_ylabel('Feature importance')

    # Format x axis labels
    xtick_labels = [data_dict.loc[label, 'Name'] for label in feature_importances.index]
    ax.set_xticklabels(xtick_labels)

    # Return result
    return ax
