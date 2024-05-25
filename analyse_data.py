# Import necessary libraries
import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Constants
HEAD_INDEX = 0
LEFT_FIN = 1
RIGHT_FIN = 2
BODY_END = 3
BACK_FIN = 4
MIDDLE = 5


# Function to load the HDF5 file and extract data
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

# Function to check for missing values (NaNs)
def check_nans(data):
    """
    Check for missing values in the data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        int: Number of NaN values in the data.
    """
    return np.isnan(data).sum()

# Function to fill missing values
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

# Function to plot the head locations of the fish
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
    plt.plot(-1 * node_loc[:, 1, 0], 'y')
    plt.plot(-1 * node_loc[:, 1, 1], 'g')
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

# Main execution block
if __name__ == "__main__":
    # Define the path to the HDF5 file
    h5_file_path = '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_01/poses/20230316_Mormyrus_Pair_01.000_20230316_Mormyrus_Pair_01.analysis.h5'

    # Ensure the file exists
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

    # Load the HDF5 file
    tracks_data = extract_locations(h5_file_path)

    # Check for missing values
    initial_nans = check_nans(tracks_data)
    print("Initial number of NaN values:", initial_nans)
    tracks_data_filled = fill_missing(tracks_data)
    final_nans = check_nans(tracks_data_filled)
    print("Number of NaN values after filling:", final_nans)

    # Extract head locations
    head_locations = tracks_data_filled[:, HEAD_INDEX, :, :]

    # Plot the head locations
    plot_node_locations(head_locations)
