import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load raw data from the provided file path.
    """
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    """
    Handle missing values in the dataset:
    - Fill missing numerical values with the median.
    - Fill missing categorical values with the mode.
    """
    for column in data.select_dtypes(include=np.number).columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    for column in data.select_dtypes(include='object').columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def preprocess_data(data):
    """
    Perform additional preprocessing such as scaling features.
    """
    from sklearn.preprocessing import StandardScaler

    features = data.drop(columns=['risk_category'])  # Assuming 'risk_category' is the target
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Re-attach the target column
    data[features.columns] = scaled_features
    return data

if __name__ == "__main__":
    # Example usage
    file_path = 'data/raw/battery_data.csv'
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = preprocess_data(data)
    
    # Save the processed data
    data.to_csv('data/processed/processed_data.csv', index=False)
    print("Data preprocessing completed and saved to 'data/processed/processed_data.csv'")
