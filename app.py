# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and dataset
model = joblib.load("rating_predictor.pkl")
df = pd.read_csv("google_review_ratings.csv")
df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

st.set_page_config(page_title="ReviewInsight", layout="wide")
st.title("📊 ReviewInsight: Google Review Ratings Analyzer")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Home", "Data Preview", "Visualizations", "Predictor", "Report"])

# === Home Page ===
if menu == "Home":
    st.markdown("### Welcome to ReviewInsight!")
    st.write("This app allows you to:")
    st.markdown("""
    - 🧭 Explore the Google review ratings dataset  
    - 📊 Visualize review patterns across different categories  
    - 🔮 Predict the value of **Category 24** using other review categories  
    - 🧾 View model performance metrics
    """)

# === Data Preview ===
elif menu == "Data Preview":
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

# === Visualizations ===
elif menu == "Visualizations":
    st.subheader("📈 Visual Explorations")
    
    st.write("### Ratings Distribution (All Categories)")
    fig, ax = plt.subplots(figsize=(10, 4))
    df.hist(ax=ax, bins=15, figsize=(15, 8), layout=(4, 6))
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax2)
    st.pyplot(fig2)

# === Predictor ===
elif menu == "Predictor":
    st.subheader("🔮 Predict Category 24 (Google User Rating)")

    with st.form("prediction_form"):
        user_input = []
        cols = df.columns[:-1]  # All columns except Category 24

        for col in cols:
            val = st.number_input(f"{col}", min_value=0.0, max_value=5.0, step=0.1, value=2.5)
            user_input.append(val)

        submitted = st.form_submit_button("Predict")

        if submitted:
            prediction = model.predict([user_input])[0]
            st.success(f"📌 Predicted Category 24 Rating: **{round(prediction, 2)}**")

# === Report ===
elif menu == "Report":
    st.subheader("📊 Model Performance Report")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    X = df.drop(columns=["Category 24"])
    y = df["Category 24"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.metric("R² Score", f"{r2:.4f}")
    st.metric("Mean Squared Error", f"{mse:.4f}")

    st.write("### Scatter Plot: Actual vs Predicted")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.5)
    ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    ax3.set_title("Actual vs Predicted Category 24")
    st.pyplot(fig3)
