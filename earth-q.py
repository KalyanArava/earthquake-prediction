# ==========================================
# Earthquake Prediction System
# UI Refactored + Advanced Features
# Final Year Project (V2)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(page_title="Earthquake Prediction", layout="wide")

# ------------------------------------------
# Custom CSS (Fonts + Colors)
# ------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
h1 { color: #1f4fd8; }
.stMetric { background-color: #f4f7ff; padding: 15px; border-radius: 10px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Earthquake Prediction System")
st.caption("Final Year Project – ML + Streamlit")

# ------------------------------------------
# Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("database.csv")
    df = df[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]

    timestamps = []
    for d, t in zip(df["Date"], df["Time"]):
        try:
            ts = datetime.datetime.strptime(d + " " + t, "%m/%d/%Y %H:%M:%S")
            timestamps.append(ts.timestamp())
        except:
            timestamps.append(np.nan)

    df["Timestamp"] = timestamps
    df.dropna(inplace=True)
    df.drop(["Date", "Time"], axis=1, inplace=True)
    return df

data = load_data()

# ------------------------------------------
# Prepare Model
# ------------------------------------------
X = data[["Timestamp", "Latitude", "Longitude"]]
y = data[["Magnitude", "Depth"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(16, activation="relu"),
    Dense(2)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# ------------------------------------------
# Session State
# ------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ------------------------------------------
# Tabs
# ------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🗺 Risk Map", "ℹ️ Help"])

# ==========================================
# TAB 1: Prediction
# ==========================================
with tab1:
    with st.form("predict_form"):
        latitude = st.number_input("Latitude", -90.0, 90.0, 20.0, help="North–South location")
        longitude = st.number_input("Longitude", -180.0, 180.0, 80.0, help="East–West location")
        date = st.date_input("Date", help="Date of prediction")
        time_input = st.time_input("Time", help="Time of prediction")
        submit = st.form_submit_button("🔮 Predict")

    if submit:
        ts = datetime.datetime.combine(date, time_input).timestamp()
        scaled = scaler.transform([[ts, latitude, longitude]])
        pred = model.predict(scaled)
        mag, depth = float(pred[0][0]), float(pred[0][1])
        st.session_state.prediction = {"mag": mag, "depth": depth, "lat": latitude, "lon": longitude}

    if st.session_state.prediction:
        mag = st.session_state.prediction["mag"]
        depth = st.session_state.prediction["depth"]

        st.subheader("📊 Prediction Result")
        st.metric("Magnitude", f"{mag:.2f}")
        st.metric("Depth (km)", f"{depth:.2f}")

        # Safety + Explanation
        if mag < 4:
            st.success("Low Risk – Usually not dangerous")
            tips = "Stay calm. No special action needed."
        elif mag < 6:
            st.warning("Medium Risk – Can cause damage")
            tips = "Stay indoors. Move away from windows."
        else:
            st.error("High Risk – Dangerous earthquake")
            tips = "Drop, Cover, Hold On. Evacuate if needed."

        st.info(f"🧠 **Explanation:** Magnitude shows strength. Depth shows how deep it starts.\n\n🛡 **Safety Tip:** {tips}")

        # PDF Report
        if st.button("📄 Download Report"):
            file_path = "prediction_report.pdf"
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            content = [
                Paragraph("Earthquake Prediction Report", styles['Title']),
                Paragraph(f"Magnitude: {mag:.2f}", styles['Normal']),
                Paragraph(f"Depth: {depth:.2f} km", styles['Normal']),
                Paragraph(f"Safety Advice: {tips}", styles['Normal'])
            ]
            doc.build(content)
            with open(file_path, "rb") as f:
                st.download_button("⬇ Download PDF", f, file_name="earthquake_report.pdf")

# ==========================================
# TAB 2: Risk Map
# ==========================================
with tab2:
    st.subheader("🗺 Location-Based Risk Map")

    if st.session_state.prediction:
        lat = st.session_state.prediction["lat"]
        lon = st.session_state.prediction["lon"]
        mag = st.session_state.prediction["mag"]

        color = "green" if mag < 4 else "orange" if mag < 6 else "red"
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df)
        st.caption(f"Risk Level Color: {color.upper()}")
    else:
        st.info("Make a prediction to see map")

# ==========================================
# TAB 3: Help
# ==========================================
with tab3:
    st.markdown("""
    ### What do these terms mean?

    **Magnitude** – Strength of earthquake (higher = stronger)  
    **Depth** – How deep below Earth it starts  

    ### Important Note
    This is an **academic project**, not a real warning system.
    """)

st.caption("© Final Year Project | Earthquake Prediction")
