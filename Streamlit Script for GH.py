import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import GoogleV3
import joblib
# 1. Load your trained model and scaler
model = joblib.load('customer_price_sensitivity_model_20250420_003939.pkl')
scaler = joblib.load('scaler_20250420_003939.pkl')

# 2. Define your cost per hour
cost = 50

# Add Google API Key
google_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

# New geocoder using Google Maps
geolocator = GoogleV3(api_key=google_api_key)

# 3. Define a function to geocode an address
def get_lat_lon(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None
        
# 4. Define a function to find optimal rate
def suggest_optimal_rate(address):
    lat, lon = get_lat_lon(address)
    if lat is None:
        print("Address could not be geocoded.")
        return None
    
    #Create a range of rates to simulate
    rate_range = np.linspace(50, 400,700)
    simulation_df = pd.DataFrame({
        'rate':rate_range,
        'latitude': [lat] * len(rate_range),
        'longitude': [lon] * len(rate_range)
    })

    # Create interaction terms
    simulation_df['rate_lat'] = simulation_df['rate'] * simulation_df['latitude']
    simulation_df['rate_lon'] = simulation_df['rate'] * simulation_df['longitude']
    simulation_df['rate_squared'] = simulation_df['rate'] **2

    # Scale
    simulation_scaled = scaler.transform(simulation_df)

    # Predict win probability
    simulation_df['win_probability'] = model.predict_proba(simulation_scaled)[:, 1]

    # Calculated expected profit
    simulation_df['expected_profit'] = simulation_df['win_probability'] * (simulation_df['rate'] - cost)

    # Find the optimal rate
    optimal_row = simulation_df.loc[simulation_df['expected_profit'].idxmax()]
    optimal_rate = optimal_row['rate']
    anticipated_win_rate = optimal_row['win_probability']
    target_margin = 1 - (cost / optimal_row['rate'])
    
    return optimal_rate, anticipated_win_rate, target_margin, cost

# 5. Build Streamlit UI

st.title("📍 Profit Optimizing Rate Guide")

address = st.text_input("Enter Address")

if address:
    with st.spinner('🔎 Searching for best rate...'):
        optimal_rate, anticipated_win_rate, target_margin, cost = suggest_optimal_rate(address)
    if optimal_rate: # Only display if address was successfully geocoded
        st.success('✅ Optimization Complete!')
        
        st.write(f"🏡 Address: {address}")
        st.write(f"💰 Suggested Customer Rate: ${optimal_rate:.2f}")
        st.write(f"📈 Anticipated Win Rate: {anticipated_win_rate:.1%}")
        st.write(f"🏆 Target Margin: {target_margin:.1%}")
        st.write(f"💳 Anticipated Provider Rate: ${cost:.2f}")
    else:
        st.error("❌ Could not geocode this address. Please double-check the spelling or try adding a ZIP code.")
