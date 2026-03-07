import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
data = pd.read_csv("CarPrice_dataset.csv")

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Remove duplicates
data.drop_duplicates(inplace=True)

# One Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Save feature columns
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=32)

# Save model and scaler
model.save("car_price_model.keras")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully")