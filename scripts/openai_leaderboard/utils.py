import numpy as np
import matplotlib
import matplotlib.pyplot as plt

basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


class TileEncoder():
    def __init__(self, var_ranges, num_tiles, num_tilings):
        assert len(var_ranges) == len(num_tiles), \
            "Input variables length do not match"
        assert all(isinstance(val, int) and val > 0 for val in num_tiles), \
            "number of tiles should be an array of integers > 0"
        assert all(len(var_range) == 2 for var_range in var_ranges), \
            "variable range should be a finite numeric interval"
        assert isinstance(num_tilings, int), \
            "number of tilings should be an integer" 
        self.var_ranges = var_ranges
        self.num_tiles = num_tiles
        self.num_var = len(var_ranges)
        self.num_tilings = num_tilings
        self.var_coeff = np.zeros(self.num_var, dtype=np.float32)
        self.get_coeffs()
        self.iht_size = self.calc_iht_size()
        self.iht = IHT(self.iht_size)

    def get_coeffs(self):
        """Calculate the coefficients for scaling the inputs"""
        for i, var_range in enumerate(self.var_ranges):
            self.var_coeff[i] = np.floor(self.num_tiles[i] / np.abs(var_range[1] - var_range[0]))

    def calc_iht_size(self):
        iht_size = 1
        for num_tiles in self.num_tiles:
            iht_size *= (5*num_tiles)
        # print("Hash table index range is {}".format(iht_size*3))
        return iht_size

    def get_feature(self, values):
        assert len(values) == self.num_var, "Incorrect input length"
        new_values = np.array(values, dtype=np.float32) * self.var_coeff
        return tiles(self.iht, self.num_tilings, new_values)

def plot_var_history(var_history, labels, show_confidence=False,
                     color_list=None, x_label='', y_label='',
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
    if color_list is None:
        color_list = ['b', 'g', 'r', 'y', 'k', 'm']
    # Get mean over all runs
    var_means = np.mean(var_history, axis=1)
    fig, ax = plt.subplots()
    fig.set_size_inches(*fig_size)
    # Graph foramtting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
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