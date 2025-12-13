# ==========================================
# Earthquake Prediction System
# Final Year Project
# Streamlit + Machine Learning
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(page_title="Earthquake Prediction", layout="wide")
st.title("🌍 Earthquake Prediction System")
st.write("Final Year Project – Machine Learning with Streamlit")

# ------------------------------------------
# Load & Preprocess Data
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("database.csv")

    df = df[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]

    timestamps = []
    for d, t in zip(df["Date"], df["Time"]):
        try:
            ts = datetime.datetime.strptime(
                d + " " + t, "%m/%d/%Y %H:%M:%S"
            )
            timestamps.append(ts.timestamp())
        except:
            timestamps.append(np.nan)

    df["Timestamp"] = timestamps
    df.dropna(inplace=True)
    df.drop(["Date", "Time"], axis=1, inplace=True)

    return df

data = load_data()

# ------------------------------------------
# Map Visualization (FIXED)
# ------------------------------------------
st.subheader("📍 Earthquake Locations")
map_data = data.rename(columns={"Latitude": "lat", "Longitude": "lon"})
st.map(map_data[["lat", "lon"]])

# ------------------------------------------
# Prepare Data
# ------------------------------------------
X = data[["Timestamp", "Latitude", "Longitude"]]
y = data[["Magnitude", "Depth"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# Build & Train Model
# ------------------------------------------
model = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(16, activation="relu"),
    Dense(2)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# ------------------------------------------
# Session State (IMPORTANT)
# ------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ------------------------------------------
# Sidebar Form (NO AUTO REFRESH)
# ------------------------------------------
st.sidebar.header("🔢 Enter Input Parameters")

with st.sidebar.form("prediction_form"):
    latitude = st.number_input("Latitude", -90.0, 90.0, 20.0)
    longitude = st.number_input("Longitude", -180.0, 180.0, 80.0)
    date = st.date_input("Date")
    time_input = st.time_input("Time")

    submit = st.form_submit_button("🔮 Predict Earthquake")

# ------------------------------------------
# Prediction (ONLY ON CLICK)
# ------------------------------------------
if submit:
    dt = datetime.datetime.combine(date, time_input)
    timestamp = dt.timestamp()

    input_data = scaler.transform([[timestamp, latitude, longitude]])
    prediction = model.predict(input_data)

    st.session_state.prediction = {
        "magnitude": float(prediction[0][0]),
        "depth": float(prediction[0][1])
    }

# ------------------------------------------
# Display Prediction (PERSISTENT)
# ------------------------------------------
if st.session_state.prediction:
    mag = st.session_state.prediction["magnitude"]
    depth = st.session_state.prediction["depth"]

    st.subheader("📊 Prediction Results")
    st.metric("Predicted Magnitude", f"{mag:.2f}")
    st.metric("Predicted Depth (km)", f"{depth:.2f}")

    # Risk Level
    if mag < 4.0:
        st.success("🟢 Low Risk: Minor earthquake expected")
        impact = "minor impact"
    elif mag < 6.0:
        st.warning("🟡 Medium Risk: Moderate earthquake possible")
        impact = "moderate impact"
    else:
        st.error("🔴 High Risk: Strong earthquake potential")
        impact = "severe impact"

    depth_info = "shallow earthquake" if depth < 70 else "deep earthquake"

    # Summary
    st.info(
        f"📌 **Prediction Summary**\n\n"
        f"• Estimated magnitude **{mag:.2f}** indicates a *{impact}*\n"
        f"• Estimated depth **{depth:.2f} km** suggests a *{depth_info}*"
    )

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.markdown("👨‍🎓 Final Year Project | Earthquake Prediction using Machine Learning")
