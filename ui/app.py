import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="PathSnap UI", layout="wide")
st.title("PathSnap: Map-Matching Dashboard")

# P3's API URL (Change this to P3's laptop IP)
API_URL = "http://localhost:8000/predict" 

uploaded_file = st.sidebar.file_uploader("Upload .track file", type=["track", "txt"])

if uploaded_file is not None:
    # Read the uploaded file for display
    df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    
    if st.button("Reconstruct Route"):
        with st.spinner("Calling API..."):
            # Send to P3's API
            files = {'file': uploaded_file.getvalue()}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Route Reconstructed!")
                    
                    # Display Map (P3 must return lat/lon coords of the route)
                    # I am mocking this until P3 is done
                    m = folium.Map(location=[df['lat'].iloc[0], df['lon'].iloc[0]], zoom_start=15)
                    
                    # Plot Raw GPS
                    for _, row in df.iterrows():
                        folium.CircleMarker([row['lat'], row['lon']], radius=2, color='blue').add_to(m)
                    
                    folium_static(m)
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")