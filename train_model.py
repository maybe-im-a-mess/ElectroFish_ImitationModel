import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prepare_data import trainX, trainY, testX, testY


def create_lstm_model(input_shape, num_lstm_units):
    """
    Function to create an LSTM model.
    
    Args:
    - input_shape (tuple): Shape of the input data.
    - num_lstm_units (int): Number of LSTM units in each LSTM layer.
    
    Returns:
    - tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(num_lstm_units, return_sequences=True, input_shape=input_shape),
        LSTM(num_lstm_units, return_sequences=True),
        Dense(input_shape[1], activation='relu')  # Output shape matches input shape
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == "__main__":
    input_shape = (trainX.shape[1], trainX.shape[2])

    # Create and compile the model
    model = create_lstm_model(input_shape, 64)

    # Train the model
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.2)
