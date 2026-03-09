import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prepare_data import X, Y, X_train, X_test, Y_train, Y_test, X_old, X_new, Y_old, Y_new
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import ks_2samp
from river import drift


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
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
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


def monitor_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return accuracy, precision, recall


def ks_test(feature_old, feature_new):
    statistic, p_value = ks_2samp(feature_old, feature_new)
    print(f"KS Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    return statistic, p_value


def plot_feature_distribution(feature_old, feature_new, feature_name):
    plt.figure(figsize=(12, 6))
    plt.hist(feature_old, bins=30, alpha=0.5, label='Old Data')
    plt.hist(feature_new, bins=30, alpha=0.5, label='New Data')
    plt.title(f"Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def detect_drift(y_true, y_pred):
    ddm = drift.DDM()
    for i in range(len(y_true)):
        ddm.update(y_true[i] == y_pred[i])
        if ddm.change_detected:
            print(f"Change detected at index {i}")
        if ddm.warning_detected:
            print(f"Warning detected at index {i}")


if __name__ == "__main__":
    model = load_model('LSTM_original')

    predicted_trajectory = predict_trajectory(model, X_test)

    fish1_head_true = Y_test[:, :2] # Columns 0 and 1 for Fish 1 head X and Y
    fish1_head_pred = predicted_trajectory[:, :2]

    fish2_head_true = Y_test[:, 12:14]  # Columns 12 and 13 for Fish 2 head X and Y
    fish2_head_pred = predicted_trajectory[:, 12:14]

    # Plot the extracted head positions
    plot_head_positions(fish1_head_true, fish1_head_pred, fish2_head_true, fish2_head_pred)

    # Monitor performance of the model
    y_pred_old = model.predict(X_old)
    y_pred_new = model.predict(X_new)
    np.random.seed(42)
    y_true_old = np.random.randint(2, size=len(Y_old))
    y_pred_old = np.random.randint(2, size=len(Y_old))
    y_true_new = np.random.randint(2, size=len(Y_new))
    y_pred_new = np.random.randint(2, size=len(Y_new))

    accuracy_old, precision_old, recall_old = monitor_performance(y_true_old, y_pred_old)
    accuracy_new, precision_new, recall_new = monitor_performance(y_true_new, y_pred_new)

    # Statistical test for feature distribution
    feature_old = X_old[:, 0, 0]
    feature_new = X_new[:, 0, 0]
    statistic, p_value = ks_test(feature_old, feature_new)
    statistic, p_value = ks_test(feature_old, feature_new)

