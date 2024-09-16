import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform




def plot_heirarchical_clusters(cmat, data_dict, ax, **kwds):

    # Ensure diagonal 
    corr = cmat.to_numpy()
    corr = (corr + corr.T) / 2.
    np.fill_diagonal(corr, 1.)

    # Handle NaN values
    corr = np.nan_to_num(corr)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Construct plot
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=cmat.columns.to_list(), ax=ax, leaf_rotation=90
    )
    ax.set_xticklabels(data_dict.loc[dendro["ivl"], 'Name'], rotation="vertical")
    ax.set(**kwds)
    
    # Return axis
    return ax
