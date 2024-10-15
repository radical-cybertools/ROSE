import numpy as np
from texttable import Texttable
import torch
import pandas as pd


def get_device(args):
    """

    Parameters
    ----------
    args : Argument parser object

    Returns
    -------
    device: cpu or gpu
    """
    return torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def table_printer(args):
    """
    Print the parameters of the model in a Tabular format
    Parameters
    ---------
    args: argparser object
        The parameters used for the model
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.set_precision(4)
    table.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(table.draw())

def get_Z(model):
    """

    Parameters
    ----------
    model : trained neural network model

    Returns
    -------
    pi: activation level of each layer
    z: latent neural network architecture sampled from Beta-Bernoulli processes
    threshold: number of layers learnt
    """
    Z, threshold, pi = model.structure_sampler(get_pi=True, num_samples=1)
    z = Z.mean(0).cpu().detach().numpy()
    pi = pi.cpu().detach().numpy()
    return pi, z, threshold

def plot_network_mask(ax, model, ylabel=False, sz=10):
    """

    Parameters
    ----------
    ax : Matplotlib axis object
    model : Trained neural network model
    ylabel : Flag to add y-axis label

    Returns
    -------
    cbar_ax: matplotlib axes to add colorbar

    """
    pi, z, threshold = get_Z(model)
    k_position = 50
    pi = pi.reshape(-1)
    pi[threshold:] = 0
    z = z[:, :threshold]
    scale = 15
    pi = pi * scale
    x_data = np.arange(z.shape[1])
    XX, YY = np.meshgrid(x_data, np.arange(z.shape[0]))
    table = np.vstack((XX.ravel(), YY.ravel(), z.ravel())).T
    
    df = pd.DataFrame(table)
    df.columns = ['x', 'y', 'data']

    ax.set(frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(-2.8, -(scale / 2), r"$\pi$", fontsize=16)
    ax.text(k_position-2, -2.3, r"$K^+$", fontsize=16)
    ax.text(-2.5, -0.3, r"0", fontsize=14)
    ax.text(-2.5, -1 * scale - 0.5, r"1", fontsize=14)
    ax.hlines(y=-scale - 1, xmin=-0.2, xmax=len(pi), linewidth=1, color="k", linestyle="-")
    ax.bar(np.arange(len(pi)), -1. * pi, bottom=[-1] * len(pi), color="black", width=0.7)
    ax.hlines(y=-1, xmin=-0.2, xmax=len(pi), linewidth=1, color="k")
    ax.set_xlim(-5, k_position-1)
    cbar_ax = ax.scatter(x="x", y="y", c="data", s=sz, data=df, cmap='Blues', edgecolors="k", linewidths=0.2)
    ax.set_xlabel(r"Layers", fontsize=12)
    if ylabel:
        ax.set_ylabel(r"Active Neurons", fontsize=12)
        ax.yaxis.set_label_coords(0, 0.3)
    ax.invert_yaxis()
    return cbar_ax


def plot_prediction(ax, results, legend=False):
    """

    Parameters
    ----------
    ax : Matplotlib axis object
    results : dictionary that contains testing data points and predictions with standard deviation
    legend : Flag to add legend to the plot

    Returns
    -------

    """
    var = (results['total_unc'])
    ax.plot(results['xs_t'].ravel(), results['mean'], color='k', label="mean", linewidth=1, alpha=0.7)
    ax.fill_between(results['xs_t'].ravel(), results['mean'] + var, results['mean'] - var, color='green', alpha=0.3,
                    label="std")
    ax.scatter(results['x'].ravel(), results['y'].ravel(), s=1, c='gray', alpha=0.5, marker="o", label="data points")
    ax.set_ylim([-2.5, 3])
    ax.set_xlim([-2.5, 2.5])
    ax.set_xlabel("X", fontsize=12)
    ax.text(0.25, -0.2, '')

    ax.set_xticks(np.arange(-2, 3, 2))
    ax.set_xticklabels(np.arange(-2, 3, 2))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    if legend:
        ax.scatter(results['x'].ravel(), results['y'].ravel(), s=5, c='gray', alpha=0.5, marker="o")
        ax.set_ylabel("Y", fontsize=12)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.2)
    return ax

# Check if indices in one list is available in another list
def check_uniq_indices(l1, l2):
    for l in l1:
        if l in l2:
            return True
    
    return False