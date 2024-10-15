import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_processed_data(file_path):
    """
    Load preprocessed time series data for testing and validation.
    """
    data = pd.read_csv(file_path)
    return data

def prepare_time_series_data(data, feature_columns, target_column, time_steps=10):
    """
    Prepare time series data for testing and validation.
    
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

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Parameters:
    - model: Trained LSTM model.
    - X_test, y_test: Testing data (features and target).
    
    Returns:
    - accuracy: Accuracy score.
    - confusion_matrix: Confusion matrix for the classification.
    - classification_report: Precision, recall, f1-score for each class.
    """
    # Predict the classes (binary classification: 0 or 1)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, conf_matrix, class_report

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data('data/processed/processed_data.csv')

    # Define feature columns and target column
    feature_columns = [
        'positive_electrode_viscosity', 'negative_electrode_viscosity', 'electrode_coating_weight',
        'electrode_thickness', 'electrode_alignment', 'welding_bead_size', 'lug_dimensions', 
        'moisture_content_after_baking', 'electrolyte_weight', 'formation_energy', 'aging_time', 
        'pressure', 'ambient_temperature'
    ]
    target_column = 'risk_category'  # Binary classification: 0 for low-risk, 1 for high-risk

    # Prepare time series data (with a time window of 10 steps)
    X, y = prepare_time_series_data(data, feature_columns, target_column, time_steps=10)

    # Split into training and testing sets
    split_index = int(0.8 * len(X))  # 80% for training, 20% for testing
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Load the trained model
    model = tf.keras.models.load_model('models/risk_detection_lstm_classification_model.h5')

    # Evaluate the model on the test set
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)

    # Output the evaluation results
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
