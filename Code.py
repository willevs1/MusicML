import streamlit as st

st.title("Music ML Model")

# Download latest version
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/willevs1/MusicML/main/Spotify_2024_Global_Streaming_Data.csv")

# Group by artist and sum total streams
artist_streams = df.groupby("Artist")["Total Streams (Millions)"].sum().reset_index()

# Sort the result
artist_streams = artist_streams.sort_values(by="Total Streams (Millions)", ascending=False)

# Show the top entries in Streamlit
st.title("Top Artists by Total Streams")
st.dataframe(artist_streams.head(10))

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


