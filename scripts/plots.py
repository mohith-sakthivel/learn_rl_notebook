import numpy as np
# import plot tools
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython import display


color_list = ['b', 'g', 'r', 'y', 'k', 'm']


def plot_var_history(var_history, labels, show_confidence=False,
                     color_list=color_list, x_label='', y_label='',
                     y_ticks=None, log_scale=False, fig_size=(12, 6)):
    """
    Plot the value of a variable at each episode averaged over the number
    of runs at different setting

    Arguments:
        var_history - list/array of the below shape
                      (no. of settings, no. of runs, no.episodes) or
                      (no. of runs, no.episodes)
    """
    if not isinstance(var_history, np.ndarray):
        var_history = np.array(var_history)
    if var_history.ndim == 2:
        var_history = np.expand_dims(var_history, 0)
    assert var_history.ndim == 3 or var_history.ndim == 2, "invalid input"
    # Get mean over all runs
    var_means = np.mean(var_history, axis=1)
    fig, ax = plt.subplots()
    fig.set_size_inches(*fig_size)
    # Graph foramtting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_scale:
        ax.set_yscale("log")
    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Plot values over different setting
    for plot_no in range(var_means.shape[0]):
        ax.plot(var_means[plot_no], color=color_list[plot_no],
                label=labels[plot_no])
        if show_confidence:
            # calculate the 95 percent confidence interval
            num_runs = np.size(var_means, 1)
            episodes = np.arange(1, np.size(var_means, -1), 1)
            reward_ci = 1.960 * (np.std(var_history, axis=1)/np.sqrt(num_runs))
            ax.fill_between(
                            episodes,
                            var_means[plot_no]-reward_ci[plot_no],
                            var_means[plot_no]+reward_ci[plot_no],
                            color=color_list[plot_no], alpha=0.2)
    # Enable legend
    ax.legend()


class FunctionPlot_3D():
    """
        Plots a 3-dimensional function graph
    """

    def __init__(self, values, mesh_X, mesh_Y, fig_size=(8, 8),
                 x_label='', y_label='', title='', ipython=False):
        self.ipython = ipython
        self.fig = plt.figure()
        self.fig.set_size_inches(*fig_size)
        self.ax = self.fig.gca(projection='3d')
        self.surf = self.ax.plot_surface(mesh_X, mesh_Y, values,
                                         cmap=cm.coolwarm, linewidth=1,
                                         antialiased=True)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.fig.suptitle(title)
        # Add a color legend as a bar
        self.color_bar = self.fig.colorbar(self.surf, shrink=0.5, aspect=10)

        if self.ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)

    def update(self, values, mesh_X, mesh_Y, title):
        """ Updates the existing graph with the new values """
        self.surf.remove()
        self.fig.suptitle(title)
        self.surf = self.ax.plot_surface(mesh_X, mesh_Y, values,
                                         cmap=cm.coolwarm, linewidth=1,
                                         antialiased=True)
        plt.draw()
        self.color_bar.update_normal(self.surf)
        if self.ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)


class ValuePlot_2D():
    """
        Plots a 2-dimensional graph showing variable value
        at each discrete state
    """

    def __init__(self, values, mesh_X, mesh_Y, fig_size=(10, 8),
                 x_label='', y_label='', title='', ipython=False):
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(*fig_size)
        self.grid = self.ax.pcolor(mesh_X, mesh_Y, values,
                                   edgecolors='k', linewidths=2)
        # Add a color legend as a bar
        self.color_bar = self.fig.colorbar(self.grid, shrink=0.75, aspect=10)
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        if ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)

    def update(self, values, mesh_X, mesh_Y, title=''):
        """ Updates the existing graph with the new values """
        self.grid.remove()
        self.ax.set_title(title)
        self.grid = self.ax.pcolor(mesh_X, mesh_Y, values,
                                   edgecolors='k', linewidths=2)
        plt.draw()
        self.color_bar.update_normal(self.grid)
        display.display(plt.gcf())
        display.clear_output(wait=True)
