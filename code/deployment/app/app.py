import streamlit as st 
import requests
import json

# FastAPI backend URL
api_url = "http://127.0.0.1:8000/predict"

st.title("CO2 Emissions Prediction")

model_year = st.number_input("Model Year", min_value=2000, max_value=2024, value=2024)
make = st.text_input("Make") 
model_name = st.text_input("Model")
vehicle_class = st.text_input("Vehicle Class")
engine_size = st.number_input("Engine Size", min_value=, value=)
cylinders = st.number_input("Cylinders", min_value=1, max_value=, value=)
transmission = st.text_input("Transmission")
fueal_consumption_city = st.number_input()



input_data = {
    "Model_Year": model_year, 
    "Make": make,
    "Model": model_name, 
    "Vehicle_Class": vehicle_class,
    "Engine_Size": engine_size,
    "Cylinders": cylinders,
    "Transmission": transmission,
    "Fuel_Consumption_in_City": fuel_consumption_city,
    "Smog_Level": smog_level,
}

if st.button("Predict"):
    # Send POST request to the FastAPI backend
    response = requests.post(api_url, json=input_data)
    
    if response.status_code == 200:
        # Get the prediction from the response
        prediction = response.json()["CO2_Emissions"]
        st.success(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
    else:
        st.error(f"Error: {response.status_code}")