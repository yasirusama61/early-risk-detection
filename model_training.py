import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def load_processed_data(file_path):
    """
    Load preprocessed time series data from the file path.
    """
    data = pd.read_csv(file_path)
    return data

def prepare_time_series_data(data, feature_columns, target_column, time_steps=10):
    """
    Prepare time series data for training LSTM model.
    
    Parameters:
    - data: The dataset containing features and target.
    - feature_columns: List of feature column names.
    - target_column: The name of the target column.
    - time_steps: The number of time steps to consider for the LSTM.
    
    Returns:
    - X: Time series feature set.
    - y: Target variable.
    """
    X = []
    y = []
    feature_data = data[feature_columns].values
    target_data = data[target_column].values

    for i in range(time_steps, len(data)):
        X.append(feature_data[i-time_steps:i])
        y.append(target_data[i])

    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model for binary classification.
    
    Parameters:
    - input_shape: The shape of the input data.
    
    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
    
    # Compile the model with binary cross-entropy loss and accuracy as the evaluation metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val):
    """
    Train the LSTM model on the training data.
    
    Parameters:
    - X_train, y_train: Training data.
    - X_val, y_val: Validation data.
    
    Returns:
    - model: Trained LSTM model.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    return model

if __name__ == "__main__":
    # Load the processed time series data
    data = load_processed_data('data/processed/processed_data.csv')
    
    # Define feature columns and target column
    feature_columns = [
        'voltage', 'current', 'temperature', 'soc', 'soh', 
        'internal_resistance', 'cycle_count', 'discharge_time', 
        'charge_time', 'formation_energy', 'aging_time', 
        'ambient_temperature', 'pressure', 'liquid_level'
    ]
    target_column = 'risk_category'  # Binary classification: 0 for low-risk, 1 for high-risk
    
    # Prepare the time series data (with a time window of 10 steps)
    X, y = prepare_time_series_data(data, feature_columns, target_column, time_steps=10)

    # Split into training and validation sets
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    # Train the LSTM model
    model = train_lstm_model(X_train, y_train, X_val, y_val)
    
    # Save the trained model
    model.save('models/risk_detection_lstm_classification_model.h5')
    print("Model training completed and saved to 'models/risk_detection_lstm_classification_model.h5'")
    
    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
