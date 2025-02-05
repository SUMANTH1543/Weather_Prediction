import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the saved model using joblib
model = joblib.load('weather_forecast_model.pkl')

# Function to make predictions
def predict_weather(input_data):
    # Assuming input_data is a list of 10 days data with 3 features each (Temperature, Humidity, Wind Speed)
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Reshape input data to be 3D for model prediction (samples, time_steps, features)
    input_data_reshaped = input_data_scaled.reshape((1, 10, 3))  # 1 sample, 10 time steps, 3 features
    
    # Predict the weather (temperature) for the next 3 days
    prediction = model.predict(input_data_reshaped)
    
    return prediction[0]  # Return the 3-day prediction

# Streamlit UI
st.title('Real-Time Weather Prediction App')

# Interactive input fields for 10 days of data (Temperature, Humidity, Wind Speed)
st.subheader("Enter data for the last 10 days:")

# Input for 10 days of data
temperature = [st.slider(f'Temperature for day {i+1} (°C)', min_value=-50.0, max_value=50.0, value=20.0) for i in range(10)]
humidity = [st.slider(f'Humidity for day {i+1} (%)', min_value=0.0, max_value=100.0, value=50.0) for i in range(10)]
wind_speed = [st.slider(f'Wind Speed for day {i+1} (kph)', min_value=0.0, max_value=150.0, value=10.0) for i in range(10)]

# Combine inputs into a single list of lists (for each day)
input_data = list(zip(temperature, humidity, wind_speed))

# Displaying the entered data for user reference
st.write("Entered Data (last 10 days):")
input_df = pd.DataFrame(input_data, columns=['Temperature', 'Humidity', 'Wind Speed'])
st.write(input_df)

# Predict button for real-time prediction
if st.button('Predict Next 3 Days Temperature'):
    # Display loading spinner while the model is making predictions
    with st.spinner('Making predictions...'):
        prediction = predict_weather(input_data)
        
    # Displaying the 3-day prediction
    st.write(f'Predicted Temperature for the next 3 days:')
    for i, temp in enumerate(prediction, 1):
        st.write(f"Day {i}: {temp:.2f}°C")
    
    #  Display a chart for the predicted temperature over the next 3 days
    st.subheader("Predicted Temperature Trend for the Next 3 Days:")
    prediction_df = pd.DataFrame(prediction, columns=["Temperature"], index=[f"Day {i}" for i in range(1, 4)])
    st.line_chart(prediction_df)

