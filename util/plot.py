import matplotlib
import matplotlib.pyplot as plt


def plot_config(theme=None):

    # Font
    FPT = 9
    FSIZE = 'medium'
    FNAME = 'Arial'

    # Figure size
    FIGSIZE = [6,4]

    # Colors
    COLOR = '212121'
    colors = None
    if not theme:
        colors = matplotlib.cycler(  # type: ignore
            "color",
            [
                "44aa98",
                "ab4498",
                "332389",
                "86ccec",
                "ddcc76",
                "cd6477",
                "882255",
                "117732",
            ],
        )
    elif theme == "derisc":
        print(f"Using theme: {theme}")
        colors = matplotlib.cycler('color',["8f993e","b4bd01","555025","bbc493","efede7"])
    elif theme == "derisc_pres":
        print(f"Using theme: {theme}")
        FPT = 12
        FIGSIZE = [4,4]
        FNAME = 'Montserrat'
        COLOR = '8d8379'
        # plt.rcParams['font.weight'] = 'light'
        colors = matplotlib.cycler('color',["8f993e","b4bd01","555025","bbc493","efede7"])
    else:
        print(f"ERROR: {theme} theme not yet supported")

    # Other styling
    plt.rcParams['font.size'] = FPT
    plt.rcParams['figure.figsize'] = FIGSIZE
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.rcParams['font.family'] = FNAME
    plt.rcParams['axes.facecolor'] = 'None'
    plt.rcParams['axes.edgecolor'] = 'dimgray'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.color'] = 'lightgray'
    plt.rcParams['grid.linestyle'] = 'dashed'
    plt.rcParams['xtick.labelsize'] = FSIZE
    plt.rcParams['ytick.labelsize'] = FSIZE
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'None'
    plt.rcParams['legend.fontsize'] = FSIZE
    plt.rcParams['legend.title_fontsize'] = FSIZE
    plt.rcParams['axes.labelsize'] = FSIZE
    plt.rcParams['savefig.facecolor'] = 'None'
    plt.rcParams['savefig.edgecolor'] = 'None'
    plt.rcParams['axes.titlesize'] = FSIZE
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1
    plt.rc('axes', prop_cycle=colors)


def get_luminance(r, g, b):
    return 255.*(r*0.299 + g*0.587 + b*0.114)


def get_contrast_color(r, g, b):

    # Default color choices
    light_color, dark_color = "w", "#212121"
    # Luminance threshold
    L = 186

    # Determine font color based on contrast
    contrast_color = (
        dark_color if get_luminance(r, g, b) > L else light_color
    )

    # Return result
    return contrast_color


def add_bar_labels(ax):

    # Load configuration
    # plot_config()

    # Default color choices
    light_color, dark_color = "w", "#212121"
    # Luminance threshold
    L = 186

    # Add labels for each container set
    for _, c in enumerate(ax.containers):
        # Get bar color
        r, g, b, a = c.patches[0].get_facecolor()
        # Determine font color based on contrast
        font_color = get_contrast_color(r, g, b)
        labels = [f"{x:.0%}" if x > 0.05 else "" for x in c.datavalues]
        # Add labels
        ax.bar_label(
            c,
            labels=labels,
            label_type="center",
            fontsize="small",
            color=font_color
        )

    # Return result
    return ax
