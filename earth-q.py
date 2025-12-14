# ==========================================
# Earthquake Prediction System
# UI Refactored + Prediction History Graph
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
# Custom CSS (Fonts + Colors)
# ------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

h1 { color: #1f4fd8; font-weight: 600; }
h2, h3 { color: #0b5394; }

.stMetric {
    background-color: #f4f7ff;
    padding: 15px;
    border-radius: 10px;
}

.stAlert { border-radius: 10px; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

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
# Session State
# ------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------
# Tabs Layout
# ------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Prediction History", "ℹ️ About"])

# ==========================================
# TAB 1: Prediction
# ==========================================
with tab1:
    st.subheader("🔢 Enter Input Parameters")

    with st.form("prediction_form"):
        latitude = st.number_input("Latitude", -90.0, 90.0, 20.0)
        longitude = st.number_input("Longitude", -180.0, 180.0, 80.0)
        date = st.date_input("Date")
        time_input = st.time_input("Time")

        submit = st.form_submit_button("🔮 Predict Earthquake")

    if submit:
        dt = datetime.datetime.combine(date, time_input)
        timestamp = dt.timestamp()

        input_data = scaler.transform([[timestamp, latitude, longitude]])
        prediction = model.predict(input_data)

        mag = float(prediction[0][0])
        depth = float(prediction[0][1])

        st.session_state.prediction = {"mag": mag, "depth": depth}
        st.session_state.history.append({
            "Time": datetime.datetime.now(),
            "Magnitude": mag,
            "Depth": depth
        })

    if st.session_state.prediction:
        mag = st.session_state.prediction["mag"]
        depth = st.session_state.prediction["depth"]

        st.subheader("📊 Prediction Results")

        st.metric(
            "Predicted Magnitude ℹ️",
            f"{mag:.2f}",
            help="Magnitude measures the strength of an earthquake"
        )

        st.metric(
            "Predicted Depth (km) ℹ️",
            f"{depth:.2f}",
            help="Depth indicates how deep the earthquake originates"
        )

        if mag < 4.0:
            st.success("🟢 Low Risk: Minor earthquake expected")
            impact = "minor impact"
        elif mag < 6.0:
            st.warning("🟡 Medium Risk: Moderate earthquake possible")
            impact = "moderate impact"
        else:
            st.error("🔴 High Risk: Strong earthquake potential")
            impact = "severe impact"

        depth_info = "shallow" if depth < 70 else "deep"

        st.info(
            f"📌 **Prediction Summary**\n\n"
            f"• Estimated magnitude **{mag:.2f}** indicates a *{impact}*\n"
            f"• Estimated depth **{depth:.2f} km** suggests a *{depth_info} earthquake*"
        )

        st.success(
            f"🧠 **What this means:** A {impact} earthquake may be felt. "
            f"Since the earthquake is {depth_info}, surface impact could be "
            f"{'higher' if depth < 70 else 'lower'}."
        )

# ==========================================
# TAB 2: Prediction History Graph
# ==========================================
with tab2:
    st.subheader("📈 Prediction History")

    if len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("Time")["Magnitude"])
        st.line_chart(hist_df.set_index("Time")["Depth"])
    else:
        st.info("No predictions made yet.")

# ==========================================
# TAB 3: About
# ==========================================
with tab3:
    st.markdown("""
    ### About This Project

    This system uses Machine Learning to estimate **earthquake magnitude and depth**
    based on historical seismic data.

    **Magnitude** indicates the strength of the earthquake.
    **Depth** indicates how deep below the Earth’s surface it originates.

    ⚠️ This is a **research-based academic project** and not a real-time earthquake warning system.
    """)

st.markdown("---")
st.caption("👨‍🎓 Final Year Project | Earthquake Prediction using Machine Learning")
