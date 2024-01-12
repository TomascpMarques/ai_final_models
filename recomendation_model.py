import random

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Concatenate

user_classifiers = pd.read_csv('user_classifiers_data.csv', )
seen_products = pd.read_csv('seen_products_data.csv', )
bought_products = pd.read_csv('bought_products_data.csv', )

# pd.DataFrame(df['Temperature Measurements'].apply(eval).tolist(), columns=[f'Temp_{i+1}' for i in range(10)])

# Load and preprocess the generated data

users_data = {
    'user_id': user_classifiers["User ID"],
    'age': user_classifiers["Age"],
    'gender': user_classifiers["Gender"],
    'most_active_seasons': user_classifiers["Most Active Seasons"],
}
user_data = pd.DataFrame(users_data)
user_data.set_index('user_id')
user_data['user_id'] = pd.to_numeric(user_data['user_id'], errors='coerce')

product_data = {
    # 'user_id': bought_products["Product Category"],
    'product_category': bought_products["Product Category"],
    'product_discount': bought_products["Discount"],
    'product_original_price': bought_products["Product Original Price"],
    'product_season_release': bought_products["Season Bought"],
}
product_data = pd.DataFrame(product_data)

# product_data.set_index('user_id')
# product_data['user_id'] = pd.to_numeric(product_data['user_id'], errors='coerce')

encoder = preprocessing.LabelEncoder()
enc = preprocessing.LabelEncoder()

# Data Processing
user_data['gender'] = encoder.fit_transform(user_data['gender'])  # Encode gender

print("most active seasins")
print(enc.fit_transform(user_data['most_active_seasons'])[0:9])
print(enc.fit_transform(['Spring', 'Holidays', 'Fall']))

user_data['most_active_seasons'] = encoder.fit_transform(user_data['most_active_seasons'])

product_data['product_category'] = encoder.fit_transform(
    product_data['product_category'])  # Encode product_category

print("Product encoded categories: ")
print(product_data['product_category'])
print(encoder.inverse_transform(product_data['product_category']))


merged_data = user_data.join(product_data)

print("Merged Data: ", merged_data.to_csv()[0:2])

# Feature Engineering
# Calculate personalized product discount based on user's history
merged_data['personalized_discount'] = merged_data.groupby('product_category')['product_discount'].transform('mean')

# Determine the original price considering similar products in the user's history
merged_data['original_price'] = merged_data.groupby('product_category')['product_original_price'].transform('mean')

# Identify the season for product release based on user's most active seasons
merged_data['recommended_season'] = merged_data.groupby('product_category')['most_active_seasons'].transform(
    lambda x: x.mode()[0])

# Prepare input features and target
X = merged_data[['age', 'gender', 'most_active_seasons']].values
y = merged_data[['product_category', 'personalized_discount', 'original_price', 'recommended_season']].values
print(X.shape)

print(X)

# Normalize numerical features
scaler = StandardScaler()
X[:, 1:4] = scaler.fit_transform(X[:, 1:4])

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)

print("SHAPE")
print(X_train.shape[1])

# Build the neural network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='linear'),
])
# 4 output nodes for product_category, discount, original_price, season_release

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting (training lol)
model.fit(X_train, y_train, epochs=300, batch_size=42, validation_data=(X_test, y_test))

model.save('product_recommendation_model.keras')

print("ENCODED")
print(encoder.fit_transform(['Spring', 'Holidays', 'Fall']))
predict_data_in = [
    62, 0, 5,
]

# Use the trained model to make recommendations
user_input = np.array([predict_data_in])
user_input[:, 1:4] = scaler.transform(user_input[:, 1:4])

prediction = model.predict(user_input)
prediction = np.array(prediction).flatten()

print("PREDICTION")
print(prediction)

# Extracting recommendations
recommended_product_category = prediction[0]
recommended_discount = prediction[1]
recommended_original_price = prediction[2]
recommended_season_release = prediction[3]

# Print or use the recommendations as needed
print("product category:", recommended_product_category)
print("discount:", recommended_discount)
print("original price:", recommended_original_price)
print("season release:", recommended_season_release)
