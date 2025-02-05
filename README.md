# 🌦️ Weather Prediction using LSTM

📜 Project Description:
This project focuses on predicting weather conditions, specifically temperature for the next three days, using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical weather data and deployed using Streamlit for an interactive user interface. The prediction process involves scaling input data, feeding it into the LSTM model, and displaying the predicted temperature.
The project’s goal is to provide real-time weather forecasts by analyzing patterns in historical weather data.

🚀 Features:
Predict temperature for the next three days based on historical data.
Streamlit-based UI for interactive input and visualization of predictions.
Uses LSTM (Long Short-Term Memory) for time-series forecasting.
MinMaxScaler ensures input data is scaled for improved prediction accuracy.

📊 Dataset
The dataset used for this project contains historical weather data with features such as:

Temperature (in Celsius)
Humidity (%)
Wind Speed (km/h)
Data Format Example:

Date	    Temperature	  Humidity	 Wind Speed
2025-01-01	 25	          65	       10       
2025-01-02	 26	          70	       12 
Preprocessing Steps:
Remove null values and irrelevant columns.
Convert date-time values into proper time-series format.
Normalize features using MinMaxScaler for better model performance.


🛠️ Tech Stack:
Python
TensorFlow – for building the LSTM model
NumPy & Pandas – for data manipulation
scikit-learn – for preprocessing (MinMaxScaler)
Streamlit – for building the web interface
Matplotlib & Seaborn – for data visualization


📂 Project Workflow:
# Step 1: Data Preprocessing
Load and clean the dataset.
Apply MinMaxScaler to normalize the data.
Prepare sequences for training the LSTM model.
# Step 2: Model Building
Build an LSTM model with the following architecture:
Input layer
LSTM layers
Dense (fully connected) layers for output prediction
Compile the model using Mean Squared Error (MSE) as the loss function and Adam optimizer.
# Step 3: Model Training
Train the LSTM model on the historical weather data.
Save the trained model as weather_forecast_model.h5.
Fit and save the MinMaxScaler as scaler.pkl.
# Step 4: Model Deployment with Streamlit
Create an interactive UI using Streamlit to upload historical weather data.
Use the LSTM model to predict the next three days of temperature.
Display predictions in a user-friendly format with visualizations.

📌 Folder Structure
weather_prediction/
│
├── data/                  # Dataset folder
│   └── weather_data.csv   # Historical weather data
│
├── models/                # Model folder
│   ├── weather_forecast_model.pkl  # Trained LSTM model
│   └── scaler.pkl         # Saved MinMaxScaler
│
├── app.py                 # Streamlit app script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation


# 🧑‍💻 How to Run the Project
Prerequisites:
Python 3.8 or above
Required packages (listed in requirements.txt)

# Step 1: Clone the Repository
git clone https://github.com/SUMANTH1543/Weather_Prediction.git
cd Weather_Prediction

# Step 2: Install Dependences
pip install -r requirements.txt

# Step 3: Run the Streamlit App
streamlit run app.py

# Step 4: Upload Your Data and Predict
Upload a CSV file with columns Temperature, Humidity, and Wind Speed.
Click Predict to generate temperature predictions for the next three days.
# 📈 Results
The model predicts temperature with reasonable accuracy for short-term forecasting.
Predictions are displayed with visualizations for easy interpretation.
The interface is user-friendly and interactive.


# 🔮 Future Enhancements
Add support for additional weather parameters (e.g., precipitation, pressure).
Use GRU (Gated Recurrent Unit) for comparison with LSTM.
Improve accuracy with more extensive datasets.
Deploy the app on Heroku or AWS for wider accessibility.

# 🏆 Conclusion
This project demonstrates how deep learning can be applied to real-world time-series forecasting tasks like weather prediction. By combining the power of LSTM and an intuitive web interface, it provides valuable insights for users in a practical, interactive format.


