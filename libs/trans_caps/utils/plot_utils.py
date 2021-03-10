import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


def scatter(X, labels=None, ax=None, colors=None, **kwargs):
    ax = ax or plt.gca()
    # ax.set_xticks([])
    # ax.set_yticks([])
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], facecolor='k',
                   edgecolor=[0.2, 0.2, 0.2], **kwargs)
        return None
    else:
        ulabels = np.sort(np.unique(labels))
        colors = cm.rainbow(np.linspace(0, 1, len(ulabels))) \
            if colors is None else colors
        # colors = ['black', 'r', 'gold', 'green', 'darkblue', 'purple', 'hotpink', 'saddlebrown', 'chartreuse', 'aqua']
        for (l, c) in zip(ulabels, colors):
            ax.scatter(X[labels == l, 0], X[labels == l, 1], color=c,
                       edgecolor=c * 0.6, **kwargs)
        return ulabels, colors


def draw_ellipse(pos, cov, ax=None, **kwargs):
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = np.sqrt(s) / 150
    for nsig in range(1, 5):
        ax.add_patch(Ellipse(pos, nsig * width, nsig * height, angle, alpha=0.5 / nsig, **kwargs))


def scatter_mog(X, labels, mu, cov, ax=None, colors=None):
    ax = ax or plt.gca()
    ulabels, colors = scatter(X, labels=labels, ax=ax, colors=colors, zorder=10)
    for i, l in enumerate(ulabels):
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])


def plot(x, y, mu, var):
    bs, n_cls, dim = x.shape
    points = np.zeros((bs, dim))
    for i in range(x.shape[0]):
        points[i] = x[i, y[i]]

    fig, axes = plt.subplots(2, dim // 4, figsize=(dim, 5))
    lines = []
    labels = []

    dims = np.arange(dim).reshape(-1, 2)
    for i, ax in enumerate(axes.flatten()):
        dims_i = dims[i]
        x_i = points[:, dims_i]
        mu_i = mu[:, dims_i]
        var_i = var[:, dims_i]
        I = np.eye(2, 2)
        cov_i = np.expand_dims(var_i, -1) * np.expand_dims(I, 0)
        scatter_mog(x_i, y, mu_i, cov_i, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    fig.legend(lines, labels, loc='upper center')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

