import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prepare_data import X, Y, X_train, X_test, Y_train, Y_test

def create_lstm_model(input_shape, output_shape, num_lstm_units):
    """
    Function to create an LSTM model.
    
    Args:
    - input_shape (tuple): Shape of the input data.
    - num_lstm_units (int): Number of LSTM units in each LSTM layer.
    
    Returns:
    - tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(num_lstm_units, activation='relu', input_shape=input_shape),
        Dense(output_shape)  
    ])
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)  # Adjust learning rate
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

def predict_trajectory(model, input_data):
    """
    Predict the trajectory of the fish.

    Args:
        model (keras.Model): Trained LSTM model.
        input_data (np.ndarray): Input data for prediction.

    Returns:
        np.ndarray: Predicted trajectory of the fish.
    """
    predictions = model.predict(input_data)
    return predictions

def plot_head_positions(fish1_true, fish1_pred, fish2_true, fish2_pred):
    """
    Plot the head positions of the fish.

    Args:
        fish1_true (np.ndarray): Ground truth head positions of fish 1.
        fish1_pred (np.ndarray): Predicted head positions of fish 1.
        fish2_true (np.ndarray): Ground truth head positions of fish 2.
        fish2_pred (np.ndarray): Predicted head positions of fish 2.
    """
    sns.set('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15, 6]

    plt.figure()
    
    # Plot head positions over time
    plt.plot(fish1_true[:, 0], 'y', label='Fish 1 Ground Truth X')
    plt.plot(fish1_true[:, 1], 'y--', label='Fish 1 Ground Truth Y')
    plt.plot(fish1_pred[:, 0], 'r', label='Fish 1 Prediction X')
    plt.plot(fish1_pred[:, 1], 'r--', label='Fish 1 Prediction Y')

    plt.plot(fish2_true[:, 0], 'g', label='Fish 2 Ground Truth X')
    plt.plot(fish2_true[:, 1], 'g--', label='Fish 2 Ground Truth Y')
    plt.plot(fish2_pred[:, 0], 'b', label='Fish 2 Prediction X')
    plt.plot(fish2_pred[:, 1], 'b--', label='Fish 2 Prediction Y')

    plt.legend(loc="center right")
    plt.title('Head Tracks Over Time')
    plt.xlabel('Time')
    plt.ylabel('Position')

    plt.figure(figsize=(7, 7))
    
    # Plot head positions in 2D space
    plt.plot(fish1_true[:, 0], fish1_true[:, 1], 'y', label='Fish 1 Ground Truth')
    plt.plot(fish1_pred[:, 0], fish1_pred[:, 1], 'r--', label='Fish 1 Prediction')

    plt.plot(fish2_true[:, 0], fish2_true[:, 1], 'g', label='Fish 2 Ground Truth')
    plt.plot(fish2_pred[:, 0], fish2_pred[:, 1], 'b--', label='Fish 2 Prediction')

    plt.legend()
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    plt.xticks([])
    plt.yticks([])
    plt.title('Head Tracks in 2D Space')

    plt.show()

if __name__ == "__main__":
    input_shape = (3, 24)
    output_shape = Y.shape[1]

    model = create_lstm_model(input_shape, output_shape, 50)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

    predicted_trajectory = predict_trajectory(model, X)

    fish1_head_true = Y[:, :2] # Columns 0 and 1 for Fish 1 head X and Y
    fish1_head_pred = predicted_trajectory[:, :2]  # Columns 0 and 1 for Fish 1 head X and Y

    fish2_head_true = Y[:, 12:14]  # Columns 12 and 13 for Fish 2 head X and Y
    fish2_head_pred = predicted_trajectory[:, 12:14]  # Columns 12 and 13 for Fish 2 head X and Y

    # Plot the extracted head positions
    plot_head_positions(fish1_head_true, fish1_head_pred, fish2_head_true, fish2_head_pred)
