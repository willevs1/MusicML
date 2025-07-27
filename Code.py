import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🎵 Music ML Model & Stream Predictor")

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/willevs1/MusicML/main/Spotify_2024_Global_Streaming_Data.csv")

# Show top 10 records
st.subheader("🎧 Preview of the Dataset")
st.dataframe(df.head(10))

# Drop Genre for now
df = df.drop('Genre' , axis=1)

# Encode categorical columns and save encoders
label_cols = ["Country", "Artist", "Platform Type"]
le = {}
df_encoded = df.copy()
for col in label_cols:
    le[col] = LabelEncoder()
    df_encoded[col] = le[col].fit_transform(df[col])

# Features and target
X = df_encoded.drop(columns=["Total Streams (Millions)"])
y = df_encoded["Total Streams (Millions)"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
st.subheader("📈 Model Evaluation")
st.write("MSE:", mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.write("R-Squared:", r2)

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Streams (Millions)")
ax.set_ylabel("Predicted Streams (Millions)")
ax.set_title("Actual vs Predicted Streams")
ax.grid(True)
st.pyplot(fig)

# --------------------------
# 🎯 STREAM PREDICTION TOOL
# --------------------------
st.subheader("🎛 Predict Total Streams")

# User inputs
country = st.selectbox("Country", df["Country"].unique())
artist = st.selectbox("Artist", df["Artist"].unique())
album = st.selectbox("Album", df["Album"].unique())
platform = st.selectbox("Platform Type", df["Platform Type"].unique())
release_year = st.number_input("Release Year", min_value=2000, max_value=2025, value=2023)
monthly_listeners = st.number_input("Monthly Listeners (Millions)", value=10.0)
avg_stream_duration = st.number_input("Avg Stream Duration (Min)", value=3.5)

# Build input DataFrame
input_data = pd.DataFrame({
    "Country": [country],
    "Artist": [artist],
    "Album": [album],
    "Platform Type": [platform],
    "Release Year": [release_year],
    "Monthly Listeners (Millions)": [monthly_listeners],
    "Avg Stream Duration (Min)": [avg_stream_duration],
})

# Encode with the same encoders
for col in label_cols:
    input_data[col] = le[col].transform(input_data[col])

# Predict
if st.button("🎯 Predict Total Streams"):
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Total Streams: **{prediction:,.2f} Million**")
