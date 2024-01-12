import pandas as pd
import numpy as np
from keras.src.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

# Load the CSV data
df = pd.read_csv('warehouse_data.csv', sep='\t', lineterminator='\n')

# Extract features and target variable
X_temp = pd.DataFrame(df['Temperature Measurements'].apply(eval).tolist(), columns=[f'Temp_{i + 1}' for i in range(10)])
X = pd.concat([X_temp, df[['Expected Robots', 'Actual Robots', 'Goal Fulfilment Times']]], axis=1)
y = df['Temperature Measurements'].apply(lambda x: eval(x)[0])  # Use the first temperature measurement as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor as the baseline model
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)

print(X_train_scaled[0])

# Evaluate the model
rf_score = rf_model.score(X_test_scaled, y_test)
print(f'Random Forest R2 Score: {rf_score}')

y = scaler.transform(np.array([[28.15, 18.57, 20.06, 26.79, 12.71, 25.03, 24.1, 29.61, 13.54, 12.93,
                                3.0, 2.0, 207]]))
x = rf_model.predict(y)

print(x)

# Build a neural network model using Keras
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
nn_score = model.evaluate(X_test_scaled, y_test)
print(f'Neural Network MSE: {nn_score}')

new_features = np.array([[28.15, 18.57, 20.06, 26.79, 12.71, 25.03, 24.1, 29.61, 13.54, 12.93,
                          3.0, 2.0, 207]])

# Standardize the new features
new_features_scaled = scaler.transform(new_features)

predicted_next_temperature = model.predict(new_features_scaled)

# Print the predicted next temperature
print(f'Predicted Next Temperature: {predicted_next_temperature.flatten()}')

# Plotting the Random Forest Regression Predictions
rf_predictions = rf_model.predict(X_test_scaled)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.title('Random Forest Regression Predictions')
plt.xlabel('True Temperatures')
plt.ylabel('Predicted Temperatures')

# Plotting the Training Loss of the Neural Network Model
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.legend()

# Show the plots
plt.show()
