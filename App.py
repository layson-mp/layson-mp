import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and feature names
model = joblib.load("churn_model.joblib")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Risk Dashboard")

# --- Sidebar: Customer Input ---
st.sidebar.header("🧍 Customer Profile")

user_input = {}
for col in feature_names:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([user_input])

# --- 1️⃣ Churn Risk Calculator ---
st.subheader("🔮 Churn Risk Prediction")

proba = model.predict_proba(input_df)[0][1]
pred_flag = int(proba >= 0.5)

st.metric("Churn Probability", f"{proba:.2%}")
st.metric("Predicted Churn", "Yes" if pred_flag == 1 else "No")

# --- 2️⃣ Probability Distribution Visualization ---
st.subheader("📊 Probability Distribution")

fig, ax = plt.subplots()
ax.hist([proba], bins=10)
ax.set_title("Predicted Churn Probability")
ax.set_xlabel("Probability")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- 3️⃣ Feature Importance Dashboard ---
st.subheader("📌 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature").head(10))
else:
    st.info("Feature importance not available for this model.")

# --- 4️⃣ What-if Scenario Simulator ---
st.subheader("🔁 What-if Scenario Simulator")

st.write("Change values in the sidebar to simulate customer behavior and see how churn risk changes.")
