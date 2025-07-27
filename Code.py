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


