# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("google_review_ratings.csv")

# Drop unnecessary column
df = df.drop(columns=["Unnamed: 25", "User"])

# Convert invalid entries
df["Category 11"] = pd.to_numeric(df["Category 11"], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Define target and features
target = "Category 1"
X = df.drop(columns=[target])
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
model = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model and columns
joblib.dump(model, "rating_predictor.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
