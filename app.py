import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing 

new_uber = pd.read_csv("data.csv")


# Load the trained machine learning model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the predict_price function
def predict_price(month, destination,product_id, name, source, surge_multiplier, icon):
    
    input_data=[[month, destination, name, product_id, source, surge_multiplier, icon]]

    return model.predict(input_data)[0]

# Create a simple Streamlit app
st.title("Uber Price Predictor")

# Define the form inputs

source_no = st.selectbox(
    "Source", ["Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District", "Haymarket Square", "North End", "North Station", "Northeastern University", "South Station", "Theatre District", "West End"])

if source_no == "Back Bay":
    source = 0

elif source_no == "Beacon Hill":
    source = 1

elif source_no == "Boston University":
    source = 2

elif source_no == 'Fenway':
    source = 3

elif source_no == "Financial District":
    source = 4

elif source_no == "Haymarket Square":
    source = 5

elif source_no == "North End":
    source = 6

elif source_no == "North Station":
    source = 7

elif source_no == "Northeastern University":
    source = 8

elif source_no == "South Station":
    source = 9

elif source_no == "Theatre District":
    source = 10

elif source_no == "West End":
    source = 11

else:
    print("Please choose correct answer")

destination_no = st.selectbox("Destination", ["North Station", "Northeastern University", "West End", "Haymarket Square", "South Station", "Fenway", "Theatre District", "Beacon Hill", "Back Bay", "North End", "Financial District", "Boston University"])


if destination_no == "North Station":
    destination = 0

elif destination_no == "Northeastern University":
    destination = 1

elif destination_no == "West End":
    destination = 2

elif destination_no == "Haymarket Square":
    destination = 3

elif destination_no == "Financial District":
    destination = 4

elif destination_no == "Haymarket Square":
    destination = 5

elif destination_no == "South Station":
    destination = 6

elif destination_no == "Fenway":
    destination = 7

elif destination_no == "Theatre District":
    destination = 8

elif destination_no == "Beacon Hill":
    destination = 9

elif destination_no == "Back Bay":
    destination = 10

elif destination_no == "Boston University":
    destination = 11

else:
    print("Please choose correct answer")

# name = st.text_input("Cab Type")

name_no = st.radio(
    "Cab Type", ["Black SUV", "Lux", "Shared", "Taxi", "UberPool", "UberX"])

if name_no == "Black SUV":
    name = 0

elif name_no == "Lux":
    name = 1

elif name_no == "Shared":
    name = 2

elif name_no == "Taxi":
    name = 3

elif name_no == "UberPool":
    name = 4

elif name_no == "UberX":
    name = 5

else:
    print("Please choose correct answer")

month = st.selectbox("Month", [0,1])
product_id = st.selectbox("Day", [1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15])
icon_no = st.radio("Weather", ["clear-day","clear-night","cloudy","fog", "partly-cloudy-day","partly-cloudy-night","rain"])

if icon_no == "clear-day":
    icon = 0

elif icon_no == "clear-night":
    icon = 1

elif icon_no == "cloudy":
    icon = 2

if icon_no == "fog":
    icon = 3

elif icon_no == "partly-cloudy-day":
    icon = 4

elif icon_no == "partly-cloudy-night":
    icon = 5

elif icon_no == "rain":
    icon = 6

else:
    print("Please choose correct answer")

surge_multiplier = st.slider("Surge Multiplier", 1.0, 3.0, step=0.1)

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    
    prediction = predict_price(month, destination, product_id, name, source, surge_multiplier, icon)
    st.write(f"The predicted Uber ride price is: ${prediction}")
