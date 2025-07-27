import streamlit as st

st.title("Music ML Model")

# Download latest version
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/willevs1/MusicML/main/Spotify_2024_Global_Streaming_Data.csv")

# Show the top entries in Streamlit
df = df.drop('Genre',axis = 1)
st.dataframe(df.head(10))

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Encode categorical variables
df_encoded = df.copy()
label_cols = ["Country", "Artist", "Album", "Platform Type"]
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Features & target
X = df_encoded.drop(columns=["Total Streams (Millions)"])
y = df_encoded["Total Streams (Millions)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.write("MSE:", mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred) 
st.write("R-Squared:", r2)


import streamlit as st
import matplotlib.pyplot as plt

# Plot setup
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Streams (Millions)")
plt.ylabel("Predicted Streams (Millions)")
plt.title("Actual vs Predicted Streams")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.grid()
plt.tight_layout()

# Display in Streamlit
st.pyplot(plt)

import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("🎧 Music Stream Predictor")

# User input fields
country = st.selectbox("Country", df["Country"].unique())
artist = st.selectbox("Artist", df["Artist"].unique())
album = st.selectbox("Album", df["Album"].unique())
platform = st.selectbox("Platform Type", df["Platform Type"].unique())
release_year = st.number_input("Release Year", min_value=2000, max_value=2025, value=2023)
monthly_listeners = st.number_input("Monthly Listeners (Millions)", value=10.0)
avg_stream_duration = st.number_input("Avg Stream Duration (Min)", value=3.5)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "Country": [country],
    "Artist": [artist],
    "Album": [album],
    "Platform Type": [platform],
    "Release Year": [release_year],
    "Monthly Listeners (Millions)": [monthly_listeners],
    "Avg Stream Duration (Min)": [avg_stream_duration],
})

# Encode labels
for col in label_cols:
    input_data[col] = le[col].transform(input_data[col])

# Make prediction
if st.button("Predict Total Streams (Millions)"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Total Streams: **{prediction:,.2f} Million**")
