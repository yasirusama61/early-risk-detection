import requests
import json
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import time

# MES API Configuration
MES_API_URL = "https://mes-system.example.com/api"  # Replace with actual MES API URL
API_TOKEN = "your_api_token"  # Replace with your MES API token, I am showing an example
HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

# Load the trained LSTM classification model
MODEL_PATH = 'models/risk_detection_lstm_classification_model.h5'
model = load_model(MODEL_PATH)

# Define the features and time steps
FEATURE_COLUMNS = [
    'voltage', 'current', 'temperature', 'soc', 'soh', 
    'internal_resistance', 'cycle_count', 'discharge_time', 
    'charge_time', 'formation_energy', 'aging_time', 
    'ambient_temperature', 'pressure', 'liquid_level'
]
TIME_STEPS = 10

# Fetch Machine Data from MES
def fetch_machine_data():
    """
    Fetch real-time machine data from the MES system.
    
    Returns:
    - machine_data: DataFrame containing machine operating parameters.
    """
    try:
        response = requests.get(f"{MES_API_URL}/fetch_machine_data", headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        machine_data = pd.DataFrame(data)
        return machine_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching machine data: {e}")
        return pd.DataFrame()

# Send Predictions to MES
def send_predictions_to_mes(machine_id, predictions):
    """
    Send the predicted risk levels back to the MES system for decision making.
    
    Parameters:
    - machine_id: The ID of the machine where the predictions are applied.
    - predictions: Risk prediction results to send to MES.
    """
    try:
        payload = {
            "machine_id": machine_id,
            "predictions": predictions.tolist()  # Convert predictions to list format
        }
        response = requests.post(f"{MES_API_URL}/send_predictions", headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        print(f"Predictions successfully sent to MES for machine ID: {machine_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending predictions to MES: {e}")

# Prepare Real-Time Data for LSTM Model
def prepare_real_time_data(machine_data, feature_columns, time_steps=TIME_STEPS):
    """
    Prepare machine data for real-time risk prediction.
    
    Parameters:
    - machine_data: DataFrame containing machine data from MES.
    - feature_columns: List of columns used as features.
    - time_steps: Number of time steps for LSTM input.
    
    Returns:
    - X_real_time: Time series data ready for LSTM model prediction.
    """
    X_real_time = []
    feature_data = machine_data[feature_columns].values
    # Only take the latest 'time_steps' records
    for i in range(time_steps, len(machine_data)):
        X_real_time.append(feature_data[i - time_steps:i])
    return np.array(X_real_time)

# Real-Time Prediction Loop
def run_real_time_prediction():
    """
    Run a real-time prediction loop that continuously fetches data from MES,
    performs risk predictions, and sends results back to MES.
    """
    print("Starting real-time prediction...")
    while True:
        # Fetch real-time data from MES
        machine_data = fetch_machine_data()

        if not machine_data.empty:
            # Assume machine ID is available in the fetched data
            machine_id = machine_data['machine_id'].iloc[0]

            # Prepare data for prediction
            X_real_time = prepare_real_time_data(machine_data, FEATURE_COLUMNS)

            # Perform predictions (classifying as high or low risk)
            if X_real_time.shape[0] > 0:  # Ensure we have data to predict on
                y_pred_prob = model.predict(X_real_time)
                y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

                # Send the predictions back to MES
                send_predictions_to_mes(machine_id, y_pred)

        # Sleep for a while before fetching the next set of data
        time.sleep(60)  # Wait for 60 seconds before the next fetch

if __name__ == "__main__":
    # Start the real-time prediction loop
    run_real_time_prediction()
