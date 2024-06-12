import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

HEAD_INDEX = 0
LEFT_FIN = 1
RIGHT_FIN = 2
BODY_END = 3
BACK_FIN = 4
MIDDLE = 5


def extract_locations(file_path):
    """
    Load the HDF5 file and extract the track data.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Track data extracted from the file.
    """
    with h5py.File(file_path, 'r') as f:
        print("Keys:", list(f.keys()))
        tracks = f["tracks"][:].T
        print("Tracks shape:", tracks.shape)
    return tracks


def check_nans(data):
    """
    Check for missing values in the data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        int: Number of NaN values in the data.
    """
    return np.isnan(data).sum()


def fill_missing(data, kind="linear"):
    """
    Fill missing values independently along each dimension after the first.

    Args:
        data (np.ndarray): Input data with missing values.
        kind (str): Interpolation method.

    Returns:
        np.ndarray: Data with filled missing values.
    """
    initial_shape = data.shape
    data = data.reshape((initial_shape[0], -1))

    for i in range(data.shape[-1]):
        y = data[:, i]
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        data[:, i] = y

    data = data.reshape(initial_shape)
    return data


def plot_node_locations(node_loc):
    """
    Plot the locations of certain part of the fish.

    Args:
        node_loc (np.ndarray): Node locations of the fish.
    """
    sns.set('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15, 6]

    plt.figure()
    plt.plot(node_loc[:, 0, 0], 'y', label='fish-0')
    plt.plot(node_loc[:, 0, 1], 'g', label='fish-1')
    plt.plot(node_loc[:, 0, 2], 'h', label='fish-2')
    plt.plot(-1 * node_loc[:, 1, 0], 'y')
    plt.plot(-1 * node_loc[:, 1, 1], 'g')
    plt.plot(-1 * node_loc[:, 1, 2], 'h')
    plt.legend(loc="center right")
    plt.title(f'Tracks')

    plt.figure(figsize=(7, 7))
    plt.plot(node_loc[:, 0, 0], node_loc[:, 1, 0], 'y', label='fish-0')
    plt.plot(node_loc[:, 0, 1], node_loc[:, 1, 1], 'g', label='fish-1')
    plt.legend()
    plt.xlim(0, 2048)
    plt.xticks([])
    plt.ylim(0, 2048)
    plt.yticks([])
    plt.title('Tracks')

    plt.show()


if __name__ == "__main__":
    h5_file_path = '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_02/poses/20230316_Mormyrus_Pair_02.000_20230316_Mormyrus_Pair_02.analysis.h5'

    tracks_data = extract_locations(h5_file_path)

    initial_nans = check_nans(tracks_data)
    print("Initial number of NaN values:", initial_nans)
    tracks_data_filled = fill_missing(tracks_data)
    final_nans = check_nans(tracks_data_filled)
    print("Number of NaN values after filling:", final_nans)

    head_locations = tracks_data_filled[:, HEAD_INDEX, :, :]

    plot_node_locations(head_locations)
