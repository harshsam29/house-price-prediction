import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Load the dataset to get feature information
data = pd.read_csv('train.csv')
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['HouseAge'] = data['YrSold'] - data['YearBuilt']
data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']

# Get feature columns
feature_cols = data.drop(columns=['SalePrice']).columns

# Streamlit app
st.title("House Price Prediction")
st.write("Enter house details to predict the sale price.")
st.write("---")

# Create input fields for a subset of key features
st.header("Input House Features")

# Numerical inputs
total_sf = st.number_input("Total Square Footage (Basement + 1st + 2nd Floor)", min_value=0, value=2000, step=100)
lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=10000, step=100)
house_age = st.number_input("House Age (Years)", min_value=0, value=20, step=1)
remod_age = st.number_input("Years Since Remodel", min_value=0, value=10, step=1)
bedrooms = st.number_input("Bedrooms Above Ground", min_value=0, value=3, step=1)
bathrooms = st.number_input("Full Bathrooms", min_value=0, value=2, step=1)

# Categorical input (example: Neighborhood)
neighborhoods = data['Neighborhood'].unique()
neighborhood = st.selectbox("Neighborhood", neighborhoods)

# Create a DataFrame for prediction
input_data = pd.DataFrame(columns=feature_cols)
input_data.loc[0] = 0  # Initialize with zeros
input_data['TotalSF'] = total_sf
input_data['LotArea'] = lot_area
input_data['HouseAge'] = house_age
input_data['RemodAge'] = remod_age
input_data['BedroomAbvGr'] = bedrooms
input_data['FullBath'] = bathrooms
input_data['Neighborhood'] = neighborhood

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# Instructions
st.write("---")
st.write("Note: This app uses a Random Forest model trained on the Kaggle House Prices dataset. Enter values for key features to get a price prediction.")