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

