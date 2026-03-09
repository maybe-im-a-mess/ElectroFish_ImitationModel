import h5py
import numpy as np
import analyse_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess_sleap_data(h5_file, window_size=3):
    """
    Preprocess SLEAP analysis data (h5 format) for an ML model.
    
    Args:
        h5_file (str): Path to the SLEAP analysis file (h5).
        window_size (int): Size of the window for sequence generation.
        fish_index (int): Index of the fish to consider.
    
    Returns:
        np.ndarray: Input data (X) and target data (Y) ready for ML model.
    """
    with h5py.File(h5_file, 'r') as f:
        # Load datasets
        tracks = f["tracks"][:].T
        tracks_data_filled = analyse_data.fill_missing(tracks)

    num_frames, num_body_parts, num_coords, num_fish = tracks_data_filled.shape

    tracks_data_filled = tracks_data_filled[:, :, :, :2]

    X = []
    Y = []

    # Loop over each frame, starting from the W-th frame
    for t in range(window_size, num_frames):
        input_sequence = []
        for w in range(t - window_size, t):
            flattened_frame = []
            for fish in range(2):
                for body_part in range(num_body_parts):
                    flattened_frame.extend(tracks_data_filled[w, body_part, :, fish])
            
            input_sequence.append(flattened_frame)

        input_sequence = np.concatenate(input_sequence)
        
        target_frame = []
        for fish in range(2):
            for body_part in range(num_body_parts):
                target_frame.extend(tracks_data_filled[t, body_part, :, fish])
        
        X.append(input_sequence)
        Y.append(target_frame)

    X = np.array(X)
    Y = np.array(Y)

    num_samples = X.shape[0]
    num_features = num_body_parts * num_coords * 2
    X = X.reshape((num_samples, window_size, num_features))

    return X, Y


def split_data(inputs, outputs, test_size=0.2, random_state=42):
    """
    Split the data into train and test subsets.
    Args:
        dataset (np.ndarray): The dataset for ML.
        train_ratio (float): Ratio of data to use for training.
    Returns:
        tuple: Train and test data ready for ML.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


def normalize_data(trainX, trainY):
    """
    Function to normalize data.
    
    Args:
    - trainX (np.ndarray): Training data.
    - trainY (np.ndarray): Testing data.
    
    Returns:
    - tuple: Normalized training and testing data, scaler used for normalization.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX_reshaped = trainX.reshape(trainX.shape[0], -1)
    trainY_reshaped = trainY.reshape(trainY.shape[0], -1)
    trainX_normalized = scaler.fit_transform(trainX_reshaped)
    trainY_normalized = scaler.transform(trainY_reshaped)
    trainX_normalized = trainX_normalized.reshape(trainX.shape)
    trainY_normalized = trainY_normalized.reshape(trainY.shape)
    return trainX_normalized, trainY_normalized, scaler


h5_files = [
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_01/poses/20230316_Mormyrus_Pair_01.000_20230316_Mormyrus_Pair_01.analysis.h5',
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_02/poses/20230316_Mormyrus_Pair_02.000_20230316_Mormyrus_Pair_02.analysis.h5',
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_03/poses/20230316_Mormyrus_Pair_03.000_20230316_Mormyrus_Pair_03.analysis.h5',
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_04/poses/20230316_Mormyrus_Pair_04.000_20230316_Mormyrus_Pair_04.analysis.h5',
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_05/poses/20230316_Mormyrus_Pair_05.000_20230316_Mormyrus_Pair_05.analysis.h5',
    '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_06/poses/20230316_Mormyrus_Pair_06.000_20230316_Mormyrus_Pair_06.analysis.h5'
]

# Preprocess each file and combine datasets
X, Y = [], []

for h5_file in h5_files:
    X_file, Y_file = preprocess_sleap_data(h5_file)
    X.append(X_file)
    Y.append(Y_file)

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

# Optionally, split the data into training and testing sets
X_train, X_test, Y_train, Y_test = split_data(X, Y)

X_old, Y_old = preprocess_sleap_data(h5_files[0])
X_new, Y_new = preprocess_sleap_data(h5_files[5])


if __name__=="__main__":
    print(X.shape)
    print(Y.shape)

