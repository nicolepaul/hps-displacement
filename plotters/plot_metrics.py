import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def plot_roc_auc(model, X_test, y_test, class_names, ax):
    if len(class_names) > 2:
        if 0 not in class_names:
            class_names = {key-1: class_names[key] for key in class_names}
        ax = plot_roc_multi(model, X_test, y_test, class_names, ax)
    else:
        ax = plot_roc_single(model, X_test, y_test, class_names, ax)
    return ax
    
def plot_roc_single(model, X_test, y_test, class_names, ax):
    # Parse inputs and get scores
    classes = list(range(len(class_names)))
    y_score = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    # Calculate ROC curves and AUC
    fpr, tpr, _ = roc_curve(y_test_bin, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    # Construct plot
    ax.plot(fpr, tpr, lw=2,
             label=f'{class_names[0]} (area = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    return ax

def plot_roc_multi(model, X_test, y_test, class_names, ax):
    # Parse inputs and get scores
    classes = list(range(len(class_names)))
    y_score = None
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        y_score = model._predict_proba_lr(X_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    colors = [f'C{i}' for i in range(n_classes)]
    # Initialize variables
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Get ROC curves and AUCs
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Construct plot
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (area = {roc_auc[i]:0.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    return ax


def plot_confusion_matx(y_test, y_pred, class_names, ax=None, norm=None):

    # Normalization options
    norms = ['pred', 'true', 'all', None]
    colors = ['YlGn', 'BuPu', 'RdPu', 'OrRd']
    fmts = ['.1%', '.1%', '.1%', ',.0f']
    titles = ["Normalized by predicted classes", "Normalized by true classes",
              "Normalized by all samples", "Not normalized"]

    # Create plots
    if not norm:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7,6))
        for i, ax in zip(range(len(norms)), axs.flatten()):
            norm, c, fmt, title = norms[i], colors[i], fmts[i], titles[i]
            cm = confusion_matrix(y_test, y_pred, normalize=norm)
            disp = ConfusionMatrixDisplay(
                                        confusion_matrix=cm,
                                        display_labels=class_names.values(),
                                        ).plot(
                                                values_format=fmt,
                                                cmap=c,
                                                include_values=True,
                                                ax=ax,
                                                colorbar=False)
            ax.set_xticklabels(class_names.values(), rotation=90)
            ax.set_title(title)
        plt.tight_layout()
        return fig, axs
    else:
        ax = ax or plt.gca()
        cm = confusion_matrix(y_test, y_pred, normalize=norm)
        disp = ConfusionMatrixDisplay(
                                    confusion_matrix=cm,
                                    display_labels=class_names.values(),
                                    ).plot(
                                            values_format='.1%',
                                            cmap='YlGn',
                                            include_values=True,
                                            ax=ax,
                                            colorbar=False)
        ax.set_xticklabels(class_names.values(), rotation=90)
        ax.set_title(titles[norms.index(norm)])
        return ax
