import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/processed_data.csv')

# Define features and target
features = [
    'positive_electrode_viscosity', 'negative_electrode_viscosity', 'electrode_coating_weight', 
    'electrode_thickness', 'electrode_alignment', 'welding_bead_size', 'lug_dimensions', 
    'moisture_content_after_baking', 'electrolyte_weight', 'formation_energy', 'aging_time', 
    'pressure', 'ambient_temperature'
]
target = 'risk_level'

# Feature scaling and target encoding
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('target', axis=1))

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(data['target'])

# Reshape data for LSTM (samples, timesteps, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(3, activation='softmax'))  # 3 output units for multi-class classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# TimeSeriesSplit setup (for Rolling Window approach)
n_splits = 5  # Define the number of splits
tscv = TimeSeriesSplit(n_splits=n_splits)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Rolling Window Cross-Validation
print("Starting Rolling Window Cross-Validation...")
fold = 1
rolling_window_scores = []

for train_index, test_index in tscv.split(X_lstm):
    print(f"Fold {fold}:")
    
    # Split the data
    X_train, X_test = X_lstm[train_index], X_lstm[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Build and train LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
    
    # Evaluate on test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    rolling_window_scores.append(accuracy)
    
    # Print results for this fold
    print(f"Accuracy for Fold {fold}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Fold {fold}')
    plt.colorbar()
    tick_marks = np.arange(len(encoder.classes_))
    plt.xticks(tick_marks, encoder.classes_, rotation=45)
    plt.yticks(tick_marks, encoder.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    fold += 1

print(f"Average Accuracy (Rolling Window): {np.mean(rolling_window_scores):.4f}")

# Expanding Window Cross-Validation
print("\nStarting Expanding Window Cross-Validation...")
expanding_window_scores = []
train_size = 0.7  # Start with 70% of the data

# Manually perform expanding window
for i in range(1, n_splits+1):
    train_size = int(i * len(X_lstm) / (n_splits+1))  # Increase training size each iteration
    X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
    y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]
    
    print(f"Expanding Window Step {i}: Train size = {train_size}, Test size = {len(X_test)}")
    
    # Build and train LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
    
    # Evaluate on test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    expanding_window_scores.append(accuracy)
    
    # Print results for this step
    print(f"Accuracy for Expanding Window Step {i}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Expanding Window Step {i}')
    plt.colorbar()
    tick_marks = np.arange(len(encoder.classes_))
    plt.xticks(tick_marks, encoder.classes_, rotation=45)
    plt.yticks(tick_marks, encoder.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print(f"Average Accuracy (Expanding Window): {np.mean(expanding_window_scores):.4f}")
