# ✈️ Flight Price Prediction Using Artificial Neural Networks (ANN)

This project leverages Artificial Neural Networks (ANN) to predict flight ticket prices based on user-provided information such as departure location, destination, date, and class. The model is trained on a cleaned dataset and deployed via a Flask web application.

## 📊 Dataset

The dataset used is the [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) dataset from Kaggle. It contains information about flights between major Indian cities, including:

* Airline
* Source City
* Destination City
* Class (Economy/Business)
* Days Left for Departure
* Price

The dataset has been cleaned and preprocessed, resulting in the `Clean_Dataset.csv` file used for training and testing the model.

## 🧠 Model Architecture

The model is a feedforward neural network implemented using TensorFlow and Keras. It consists of:

* Input Layer: Accepts encoded features.
* Hidden Layers: Two dense layers with ReLU activation and dropout for regularization.
* Output Layer: Single neuron for regression output (price prediction).

The model is compiled with the Adam optimizer and mean squared error loss function.

## 🛠️ Project Structure

```
Flight_Price_Prediction/
├── Clean_Dataset.csv
├── model.h5
├── main.py
├── index.html
├── result.html
└── README.md
```

* `Clean_Dataset.csv`: Preprocessed dataset used for training.
* `model.h5`: Saved trained model.
* `main.py`: Main application script that handles data processing, model training/loading, and Flask routes.
* `index.html` & `result.html`: web interface.

## 🚀 Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/matheusherman/Flight_Price_Prediction.git
   cd Flight_Price_Prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > *Note: If `requirements.txt` is not available, install necessary libraries manually: `flask`, `pandas`, `numpy`, `scikit-learn`, `tensorflow`, etc.*

4. **Run the application:**

   ```bash
   python main.py
   ```

   > *The app will start a local server. Visit `http://localhost:8080` in your browser to use the interface.*

## 🌐 Web Interface

The Flask web application provides a user-friendly interface:

1. Navigate to `http://localhost:8080`.
2. Fill in the flight details: origin, destination, class, and departure date.
3. Submit the form to receive predicted prices for different airlines.

## 🧠 Model Development & Training Process

The model development followed a typical machine learning workflow:

### 1. **Data Preprocessing**

* Selected relevant features: `['airline', 'source_city', 'destination_city', 'class', 'days_left', 'price']`
* Categorical variables were encoded using `LabelEncoder`.
* Features were standardized with `StandardScaler`.
* The dataset was split into training and test sets using an 80/20 ratio.

### 2. **Model Architecture**

* **Input Layer**: Accepts 5 numerical features.
* **Hidden Layers**:

  * Dense layer with 128 neurons + ReLU activation + Dropout (20%)
  * Dense layer with 64 neurons + ReLU activation + Dropout (20%)
  * Dense layer with 32 neurons + ReLU activation
* **Output Layer**: 1 neuron (regression output)

```python
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')
```

### 3. **Training Details**

* Trained for **100 epochs** with a batch size of **32**
* Validation split of **20%** during training


## 📈 Model Evaluation

The model was evaluated on the test set using three standard regression metrics:

| Metric                         | Value (example) |
| ------------------------------ | --------------- |
| MAE (Mean Absolute Error)      | \~1200 ₹        |
| MSE (Mean Squared Error)       | \~3.1e+06       |
| RMSE (Root Mean Squared Error) | \~1760 ₹        |

> These values indicate that, on average, the model predicts ticket prices within a \~₹1,200 margin of error.

## 💡 Future Improvements

* Hyperparameter tuning (layers, neurons, learning rate)
* Use embeddings for categorical variables instead of label encoding
* Experiment with other models like XGBoost or ensemble methods
* Incorporate real-time flight data through APIs


