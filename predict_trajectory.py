import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import *
from train_model import *
import matplotlib as mpl

# Constants for body parts indices
HEAD_INDEX = 0
LEFT_FIN = 1
RIGHT_FIN = 2
BODY_END = 3
BACK_FIN = 4
MIDDLE = 5

def plot_node_locations(node_loc):
    """
    Plot the locations of certain parts of the fish.

    Args:
        node_loc (np.ndarray): Node locations of the fish.
    """
    # Example usage
    initial_sequence = trainX[:10]  # Use the last 10 frames from the training data as the starting point
    num_steps_to_predict = 20

    # Assuming you have a scaler used for normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(trainX.reshape(trainX.shape[0], -1))

    model = load_model("model1")
    predicted_trajectory = predict_trajectory(model, initial_sequence, num_steps_to_predict, scaler)

    # Plot the predicted trajectory
    plt.figure(figsize=(10, 6))
    for i in range(predicted_trajectory.shape[1] // 2):  # Assuming 2D positions (X, Y)
        plt.plot(predicted_trajectory[:, 2 * i], predicted_trajectory[:, 2 * i + 1], label=f'Body Part {i+1}')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Predicted Fish Trajectory')
    plt.legend()
    plt.show()

def predict_trajectory(model, input_data):
    """
    Predict the trajectory of the fish for a certain number of time steps.

    Args:
        model (keras.Model): Trained LSTM model.
        input_data (np.ndarray): Input data for prediction.
        num_time_steps (int): Number of time steps to predict.

    Returns:
        np.ndarray: Predicted trajectory of the fish.
    """
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Example usage
    input_shape = (3, 24)
    output_shape = Y.shape[1]
    model = create_lstm_model(input_shape, output_shape, 50)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)


    predicted_trajectory = predict_trajectory(model, X)

    fish1_head_true = Y[:, :2] # Columns 0 and 1 for Fish 1 head X and Y
    fish1_head_pred = predicted_trajectory[:, :2]  # Columns 0 and 1 for Fish 1 head X and Y

    fish2_head_true = Y[:, 12:14]  # Columns 12 and 13 for Fish 2 head X and Y
    fish2_head_pred = predicted_trajectory[:, 12:14]  # Columns 12 and 13 for Fish 2 head X and Y



    # Fish 1 Head Indices
    sns.set('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15, 6]

    plt.figure()
    
    # Plot trajectories over time
    plt.plot(fish1_head_true[:, 0], 'y', label='Fish 1 Ground Truth X')
    plt.plot(fish1_head_true[:, 1], 'y--', label='Fish 1 Ground Truth Y')
    plt.plot(fish1_head_pred[:, 0], 'r', label='Fish 1 Prediction X')
    plt.plot(fish1_head_pred[:, 1], 'r--', label='Fish 1 Prediction Y')

    plt.plot(fish2_head_true[:, 0], 'g', label='Fish 2 Ground Truth X')
    plt.plot(fish2_head_true[:, 1], 'g--', label='Fish 2 Ground Truth Y')
    plt.plot(fish2_head_pred[:, 0], 'b', label='Fish 2 Prediction X')
    plt.plot(fish2_head_pred[:, 1], 'b--', label='Fish 2 Prediction Y')

    plt.legend(loc="center right")
    plt.title('Head Tracks Over Time')
    plt.xlabel('Time')
    plt.ylabel('Position')

    plt.figure(figsize=(7, 7))
    
    # Plot trajectories in 2D space
    plt.plot(fish1_head_true[:, 0], fish1_head_true[:, 1], 'y', label='Fish 1 Ground Truth')
    plt.plot(fish1_head_pred[:, 0], fish1_head_pred[:, 1], 'r--', label='Fish 1 Prediction')

    plt.plot(fish2_head_true[:, 0], fish2_head_true[:, 1], 'g', label='Fish 2 Ground Truth')
    plt.plot(fish2_head_pred[:, 0], fish2_head_pred[:, 1], 'b--', label='Fish 2 Prediction')

    plt.legend()
    plt.xlim(0, 2048)
    plt.xticks([])
    plt.ylim(0, 2048)
    plt.yticks([])
    plt.title('Head Tracks in 2D Space')

    plt.show()