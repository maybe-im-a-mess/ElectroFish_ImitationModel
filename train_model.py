import tensorflow as tf
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


if __name__ == "__main__":
    input_shape = (3, 24)
    output_shape = Y.shape[1]

    model = create_lstm_model(input_shape, output_shape, 50)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

    model.save("LSTM_original")
