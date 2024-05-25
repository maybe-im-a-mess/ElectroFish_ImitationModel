import h5py
import numpy as np
import analyse_data


def preprocess_sleap_data(h5_file, sequence_length):
    """
    Preprocess SLEAP analysis data (h5 format) for an ML model.
    Args:
        h5_file (str): Path to the SLEAP analysis file (h5).
        sequence_length (int): The length of each sequence.
    Returns:
        np.ndarray: Preprocessed data ready for ML model.
    """
    with h5py.File(h5_file, 'r') as f:
        # Load datasets
        tracks = f["tracks"][:].T  # Transpose to get the shape (num_frames, num_body_parts, num_coords, num_fish)
        tracks_data_filled = analyse_data.fill_missing(tracks)  # Assuming fill_missing is defined elsewhere
        node_names = f['node_names'][:].astype(str)

    num_frames, num_body_parts, num_coords, num_fish = tracks_data_filled.shape
    num_features_per_frame = num_body_parts * 3 * num_fish  # 6 body parts * 3 (X, Y, orientation) * 2 fish

    # Initialize the list to hold sequences
    sequences = []

    # Iterate over the frames to create sequences
    for start_idx in range(0, num_frames - sequence_length + 1):
        sequence = []
        for frame_idx in range(start_idx, start_idx + sequence_length):
            fish_positions = tracks_data_filled[frame_idx]

            frame_data = []
            for fish_idx in range(num_fish):
                fish_positions_single = fish_positions[:, :, fish_idx]
                head_index = np.where(node_names == 'head')[0][0]
                center_index = np.where(node_names == 'middle')[0][0]
                head_positions = fish_positions_single[head_index]
                fish_center = fish_positions_single[center_index]
                fish_direction = head_positions - fish_center
                fish_orientation = np.arctan2(fish_direction[1], fish_direction[0])
                fish_orientation_repeated = np.full((num_body_parts, 1), fish_orientation)
                fish_data = np.concatenate((fish_positions_single, fish_orientation_repeated), axis=1)
                frame_data.append(fish_data.flatten())

            sequence.append(np.concatenate(frame_data))

        sequences.append(sequence)

    return np.array(sequences)


def split_data(dataset, train_ratio=0.67):
    """
    Split the data into train and test subsets.
    Args:
        dataset (np.ndarray): The dataset for ML.
        train_ratio (float): Ratio of data to use for training.
    Returns:
        tuple: Train and test data ready for ML.
    """
    train_size = int(len(dataset) * train_ratio)
    train, test = dataset[:train_size], dataset[train_size:]
    return train, test


def create_and_reshape_dataset(train, test, look_back=1):
    """
    Convert train and test data into dataset matrices and reshape into X=t and Y=t+1.
    
    Args:
        train (np.ndarray): The training dataset for ML.
        test (np.ndarray): The testing dataset for ML.
        look_back (int): Number of time steps to look back.
        
    Returns:
        tuple: Train and test data ready for ML.
    """
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i: (i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

        return np.array(dataX), np.array(dataY)

    # Create dataset matrices for train and test data
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    return trainX, trainY, testX, testY


h5_file = '/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_01/poses/20230316_Mormyrus_Pair_01.000_20230316_Mormyrus_Pair_01.analysis.h5'
sequence_length = 10
preprocessed_data = preprocess_sleap_data(h5_file, sequence_length)

train, test = split_data(preprocessed_data)
trainX, trainY, testX, testY = create_and_reshape_dataset(train, test)


if __name__=="__main__":
    print("preprocessed_data", preprocessed_data.shape)
    print("train", train.shape)
    print("test", test.shape)
    print("trainX", trainX.shape)
    print("trainY", trainY.shape)
    print("testX", testX.shape)
    print("testY", testY.shape)
