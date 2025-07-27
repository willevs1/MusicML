# Download latest version
dataset_path = kagglehub.dataset_download("atharvasoundankar/spotify-global-streaming-data-2024")
print("Path to dataset files:", dataset_path)
df = pd.read_csv(os.path.join(dataset_path, os.listdir(dataset_path)[0]))
print(df.head())

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Encode categorical variables
df_encoded = df.copy()
label_cols = ["Country", "Artist", "Album", "Genre", "Platform Type"]
for col in label_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Features & target
X = df_encoded.drop(columns=["Total Streams (Millions)"])
y = df_encoded["Total Streams (Millions)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred) 
print("R-Squared:", r2)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Streams (Millions)")
plt.ylabel("Predicted Streams (Millions)")
plt.title("Actual vs Predicted Streams")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid()
plt.tight_layout()
plt.show()
