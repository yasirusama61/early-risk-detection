# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/multiclass_data.csv')

# Define the feature columns
features = [
    'positive_electrode_viscosity', 'negative_electrode_viscosity', 'electrode_coating_weight',
    'electrode_thickness', 'electrode_alignment', 'welding_bead_size', 'lug_dimensions', 
    'moisture_content_after_baking', 'electrolyte_weight', 'formation_energy', 'aging_time', 
    'pressure', 'ambient_temperature'
]

# Define the target
target = 'risk_category'  # Multi-class target: 0 - low risk, 1 - medium risk, 2 - high risk

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Normalize the feature data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the target for multi-class classification
y_categorical = to_categorical(y, num_classes=3)

# Reshape input for LSTM (samples, timesteps, features)
# Assuming we have time-series data with 1 timestep
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Build the LSTM model for multi-class classification
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes (low, medium, high risk)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.argmax(axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=['Low Risk', 'Medium Risk', 'High Risk']))

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - Multi-Class Risk Levels')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Plot model accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('models/multiclass_risk_detection_lstm.h5')
