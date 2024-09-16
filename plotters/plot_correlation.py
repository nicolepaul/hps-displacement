
import numpy as np
import matplotlib.pyplot as plt
from util.plot import plot_config, get_contrast_color


def plot_correlation_matx(corr_matx, ax, cmap=None, **kwds):

    # Load configuration
    # plot_config()

    # Determine data limits
    vmin = -1 if corr_matx.min().min() < 0 else 0
    vmax = 1

    # Create colormap
    if cmap:
        pass
    else:
        if vmin == -1:
            cmap = "bwr_r"
        else:
            cmap = "Greens"
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='silver', alpha=0.6)

    # Plot gridded values
    ax.imshow(corr_matx, cmap=cmap, vmin=vmin, vmax=vmax, **kwds)

    # Adjust labels
    ax.set_xticks(range(corr_matx.shape[0]), labels=corr_matx.index.values, rotation=90)
    ax.set_yticks(range(corr_matx.shape[0]), labels=corr_matx.index.values)

    # Add data labels
    for (j,i),val in np.ndenumerate(corr_matx):
        if val == val:
            label = f"{val:.0%}"
            r, g, b, a = cmap((val - vmin)/(vmax - vmin))
            font_color = get_contrast_color(r, g, b)
            ax.text(i, j, label, ha='center', va='center', color=font_color)

    # Return result
    return ax
