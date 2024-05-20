import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='.')
data = pd.read_csv('Clean_Dataset.csv')
data = data[['airline', 'source_city', 'destination_city', 'class', 'days_left', 'price']]
label_encoders = {}
for column in ['airline', 'source_city', 'destination_city', 'class']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model_path = 'model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # A saída é um único valor (preço)
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)
    model.save(model_path)

def predict_price(source_city, destination_city, flight_class, days_left):
    airline_prices = {}
    for airline in label_encoders['airline'].classes_:
        inputs = {
            'airline': airline,
            'source_city': source_city,
            'destination_city': destination_city,
            'class': flight_class,
            'days_left': days_left
        }

        for column in inputs.keys():
            if column in label_encoders:
                inputs[column] = label_encoders[column].transform([inputs[column]])[0]

        inputs_df = pd.DataFrame([inputs])
        inputs_scaled = scaler.transform(inputs_df)

        # Fazer a previsão
        predicted_price = model.predict(inputs_scaled)
        airline_prices[airline] = predicted_price[0][0]

    return airline_prices

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

@app.route('/')
def home():
     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        source_city = request.form['origin']
        destination_city = request.form['destination']
        flight_class = request.form['class']
        departure_date = datetime.strptime(request.form['departure_date'], '%Y-%m-%d').date()
        current_date = datetime.now().date()
        days_left = (departure_date - current_date).days

        user_input = (source_city, destination_city, flight_class, days_left)
        predicted_price = predict_price(*user_input)
        predicted_price = {k: round(v, 2) for k, v in predicted_price.items()}

        return render_template('result.html',  predicted_prices=predicted_price)

if __name__ == '__main__':
    evaluate_model(model, X_test, y_test)
    app.run(host='127.0.0.1', port=8080, debug=True)
