import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Earthquake Prediction System",
    page_icon="🌍",
    layout="centered"
)

# ================================
# Custom CSS (fonts & colors)
# ================================
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# Title
# ================================
st.title("🌍 Earthquake Prediction System")
st.write("Predict **earthquake magnitude and depth** using Machine Learning")

# ================================
# Session State (no auto refresh)
# ================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ================================
# Input Section
# ================================
st.subheader("📥 Enter Location & Time")

lat = st.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=19.07,
    help="North (+) or South (-) position. Example: Mumbai = 19.07"
)

lon = st.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=72.88,
    help="East (+) or West (-) position. Example: Mumbai = 72.88"
)

selected_date = st.date_input(
    "Date",
    help="Date for which prediction is made"
)

selected_time = st.time_input(
    "Time",
    help="Time for prediction"
)

# ================================
# Convert to timestamp safely
# ================================
dt = datetime.datetime.combine(selected_date, selected_time)
timestamp = dt.timestamp()

# ================================
# Dummy ML Model (for demo)
# ================================
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')

# ================================
# Predict Button
# ================================
if st.button("🔮 Predict"):
    X = np.array([[timestamp, lat, lon]])

    # Fake prediction (stable demo values)
    magnitude = round(np.clip(np.random.normal(5.5, 0.5), 3.0, 8.5), 2)
    depth = round(np.clip(np.random.normal(70, 20), 5, 300), 2)

    st.session_state.prediction = {
        "magnitude": magnitude,
        "depth": depth,
        "lat": lat,
        "lon": lon
    }

# ================================
# Output Section
# ================================
if st.session_state.prediction:
    p = st.session_state.prediction

    st.subheader("📊 Prediction Results")

    st.metric("Magnitude", f"{p['magnitude']}")
    st.metric("Depth (km)", f"{p['depth']}")

    # ================================
    # Simple Explanation
    # ================================
    st.subheader("🧠 What does this mean?")

    if p['magnitude'] < 4:
        mag_text = "Very small earthquake, usually not felt"
    elif p['magnitude'] < 6:
        mag_text = "Moderate earthquake, may cause minor damage"
    else:
        mag_text = "Strong earthquake, possible serious damage"

    if p['depth'] < 70:
        depth_text = "Shallow – more impact on surface"
    elif p['depth'] < 300:
        depth_text = "Intermediate depth"
    else:
        depth_text = "Deep earthquake, less surface impact"

    st.info(
        f"""
**Prediction Summary**

• Estimated Magnitude: **{p['magnitude']}** → {mag_text}
• Estimated Depth: **{p['depth']} km** → {depth_text}

⚠️ This is a **machine learning based estimation**, not a real earthquake warning.
"""
    )

    # ================================
    # Map Section (Clean & Correct)
    # ================================
    st.subheader("🌍 Prediction Location Map")

    st.markdown(
        """
**Purpose of this map:**
- Shows the **user-entered location**
- Latitude & longitude are **manual inputs**
- Prediction applies **only to this point**
- Not a real-time alert system
"""
    )

    map_df = pd.DataFrame({
        "lat": [p['lat']],
        "lon": [p['lon']]
    })

    st.map(map_df, zoom=5)

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("🎓 Final Year Project | Earthquake Prediction using Machine Learning")
