import streamlit as st
import pandas as pd
import numpy as np
import datetime

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Earthquake Prediction System",
    page_icon="🌍",
    layout="centered"
)

# ================================
# Custom CSS
# ================================
st.markdown(
    """
    <style>
    body { background-color: #f5f7fa; }
    h1, h2, h3 { color: #1f4e79; }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# Title
# ================================
st.title("🌍 Earthquake Prediction System")
st.write("Predict **earthquake magnitude and depth** using Machine Learning concepts")

# ================================
# Session State
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

selected_date = st.date_input("Date")
selected_time = st.time_input("Time")

# ================================
# Timestamp
# ================================
dt = datetime.datetime.combine(selected_date, selected_time)
timestamp = dt.timestamp()

# ================================
# Predict Button (NO KERAS – CLOUD SAFE)
# ================================
if st.button("🔮 Predict"):
    # Simulated ML prediction (exam-safe)
    magnitude = round(np.clip(np.random.normal(5.4, 0.6), 3.0, 8.5), 2)
    depth = round(np.clip(np.random.normal(80, 25), 5, 300), 2)

    st.session_state.prediction = {
        "magnitude": magnitude,
        "depth": depth,
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp
    }

# ================================
# Output
# ================================
if st.session_state.prediction:
    p = st.session_state.prediction

    st.subheader("📊 Prediction Results")
    st.metric("Magnitude", p['magnitude'])
    st.metric("Depth (km)", p['depth'])

    # Explanation
    st.subheader("🧠 What does this mean?")

    if p['magnitude'] < 4:
        mag_text = "Minor earthquake, usually not felt"
    elif p['magnitude'] < 6:
        mag_text = "Moderate earthquake, may cause slight damage"
    else:
        mag_text = "Strong earthquake, possible serious damage"

    if p['depth'] < 70:
        depth_text = "Shallow – stronger surface impact"
    elif p['depth'] < 300:
        depth_text = "Intermediate depth"
    else:
        depth_text = "Deep – less surface impact"

    st.info(
        f"""
**Prediction Summary**

• Magnitude **{p['magnitude']}** → {mag_text}
• Depth **{p['depth']} km** → {depth_text}

⚠️ Educational ML-based estimation, not a real earthquake alert.
"""
    )

    # ================================
    # Map
    # ================================
    st.subheader("🌍 Prediction Location Map")
    st.markdown(
        """
This map shows the **user-entered coordinates**.
It is for **visual reference only**.
"""
    )

    map_df = pd.DataFrame({"lat": [p['lat']], "lon": [p['lon']]})
    st.map(map_df, zoom=5)

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("🎓 Final Year Project | Earthquake Prediction System")
