import numpy as np
from sklearn.metrics import mean_squared_error
from train_model import model

def evaluate_model(model, trainX, trainY, testX, testY, scaler):
    """
    Evaluate the trained LSTM model.

    Args:
        model: Trained LSTM model.
        trainX (np.ndarray): Input data for training.
        trainY (np.ndarray): Target values for training.
        testX (np.ndarray): Input data for testing.
        testY (np.ndarray): Target values for testing.
        scaler: Scaler used for normalization.

    Returns:
        tuple: Root mean squared error for train and test datasets.
    """
    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    return trainScore, testScore


if __name__ == "__main__":
    # Load your model, train and test data, and scaler
    # Then evaluate the model
    trainScore, testScore = evaluate_model(model)