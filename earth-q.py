# ==========================================
# Earthquake Prediction System (FINAL)
# Streamlit App – Cloud Safe Version
# Final Year Project
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

# ------------------------------------------
# Custom CSS (Fonts & Colors)
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
st.caption("Final Year Project – Machine Learning + Streamlit")

# ------------------------------------------
# Load Dataset
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
        latitude = st.number_input(
            "Latitude",
            -90.0,
            90.0,
            20.0,
            help="North–South position of the location"
        )
        longitude = st.number_input(
            "Longitude",
            -180.0,
            180.0,
            80.0,
            help="East–West position of the location"
        )
        date = st.date_input("Date", help="Date of prediction")
        time_input = st.time_input("Time", help="Time of prediction")
        submit = st.form_submit_button("🔮 Predict Earthquake")

    if submit:
        ts = datetime.datetime.combine(date, time_input).timestamp()
        scaled = scaler.transform([[ts, latitude, longitude]])
        pred = model.predict(scaled)

        mag = float(pred[0][0])
        depth = float(pred[0][1])

        if mag < 4:
            risk_level = "Low Risk"
            tips = "Stay calm. No major action required."
        elif mag < 6:
            risk_level = "Medium Risk"
            tips = "Stay indoors and move away from windows."
        else:
            risk_level = "High Risk"
            tips = "Drop, Cover, Hold On. Evacuate if required."

        st.session_state.prediction = {
            "mag": mag,
            "depth": depth,
            "lat": latitude,
            "lon": longitude,
            "risk": risk_level,
            "tips": tips
        }

    if st.session_state.prediction:
        p = st.session_state.prediction

        st.subheader("📊 Prediction Result")
        st.metric("Magnitude", f"{p['mag']:.2f}")
        st.metric("Depth (km)", f"{p['depth']:.2f}")

        if p["risk"] == "Low Risk":
            st.success(p["risk"])
        elif p["risk"] == "Medium Risk":
            st.warning(p["risk"])
        else:
            st.error(p["risk"])

        st.info(
            f"🧠 **What does this mean?**\n\n"
            f"• **Magnitude** shows earthquake strength.\n"
            f"• **Depth** shows how deep it starts.\n\n"
            f"🛡 **Safety Advice:** {p['tips']}"
        )

        # ---------------- Download Report (TEXT) ----------------
        report_text = f"""
Earthquake Prediction Report
----------------------------
Latitude: {p['lat']}
Longitude: {p['lon']}

Predicted Magnitude: {p['mag']:.2f}
Predicted Depth: {p['depth']:.2f} km

Risk Level: {p['risk']}

Safety Advice:
{p['tips']}

NOTE:
This is an academic ML-based prediction,
not an official warning system.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report_text,
            file_name="earthquake_prediction_report.txt",
            mime="text/plain"
        )

# ==========================================
# TAB 2: Risk Map
# ==========================================
with tab2:
    st.subheader("🗺 Location-Based Risk Map")

    if st.session_state.prediction:
        p = st.session_state.prediction
        map_df = pd.DataFrame({"lat": [p['lat']], "lon": [p['lon']]})
        st.map(map_df)
        st.caption(f"Risk Level: {p['risk']}")
    else:
        st.info("Make a prediction to view the map")

# ==========================================
# TAB 3: Help
# ==========================================
with tab3:
    st.markdown("""
    ### 🔍 Explanation

    **Magnitude** – Strength of an earthquake  
    **Depth** – Distance below Earth surface where it starts  

    ### ⚠️ Important Note
    This application is for **academic purposes only**.
    It does NOT provide real earthquake warnings.
    """)

st.caption("© Final Year Project | Earthquake Prediction")
