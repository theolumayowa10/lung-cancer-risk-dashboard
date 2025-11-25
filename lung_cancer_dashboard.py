import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ----------------------------------------------------
# Correct path for Streamlit Deployment
# ----------------------------------------------------
DATA_PATH = "lung_cancer_examples.csv"

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}. Ensure the file exists in the repo.")
        st.stop()

df = load_data()

# ----------------------------------------------------
# Clean Data
# ----------------------------------------------------
df_clean = df.copy()
df_clean["Smokes"] = df_clean["Smokes"].astype(int)
df_clean["AreaQ"] = df_clean["AreaQ"].astype(int)
df_clean["Alkhol"] = df_clean["Alkhol"].astype(int)
df_clean["Age"] = df_clean["Age"].astype(int)
df_clean["Result"] = df_clean["Result"].astype(int)

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------
X = df_clean[["Age", "Smokes", "AreaQ", "Alkhol"]]
y = df_clean["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# Standardize for Logistic Regression
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# Train Models
# ----------------------------------------------------
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------------------------------
# Evaluate Models
# ----------------------------------------------------
log_preds = log_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)

log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

log_acc = accuracy_score(y_test, log_preds)
rf_acc = accuracy_score(y_test, rf_preds)

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(
    page_title="Lung Cancer Risk Dashboard",
    layout="wide"
)

st.title("Lung Cancer Risk — Dashboard")

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------
kpi1, kpi2 = st.columns(2)

kpi1.metric("Logistic Regression AUC", f"{log_auc:.3f}")
kpi2.metric("Random Forest Accuracy", f"{rf_acc:.3f}")

# ----------------------------------------------------
# Donut Chart — Result Distribution
# ----------------------------------------------------
fig_result = px.pie(
    df_clean,
    names="Result",
    title="Target Distribution",
    hole=0.5,
    color="Result",
    color_discrete_map={0: "#6c5ce7", 1: "#00cec9"}
)
st.plotly_chart(fig_result, use_container_width=True)

# ----------------------------------------------------
# Age Histogram
# ----------------------------------------------------
fig_age = px.histogram(
    df_clean,
    x="Age",
    nbins=20,
    title="Age Distribution"
)
st.plotly_chart(fig_age, use_container_width=True)

# ----------------------------------------------------
# Confusion Matrices
# ----------------------------------------------------
cm_log = confusion_matrix(y_test, log_preds)
cm_rf = confusion_matrix(y_test, rf_preds)

fig_cm_log = px.imshow(
    cm_log,
    text_auto=True,
    title="Logistic Regression Confusion Matrix"
)

fig_cm_rf = px.imshow(
    cm_rf,
    text_auto=True,
    title="Random Forest Confusion Matrix"
)

cm_col1, cm_col2 = st.columns(2)
cm_col1.plotly_chart(fig_cm_log, use_container_width=True)
cm_col2.plotly_chart(fig_cm_rf, use_container_width=True)

# ----------------------------------------------------
# Feature Importance (Random Forest)
# ----------------------------------------------------
importance = rf_model.feature_importances_
fig_imp = px.bar(
    x=importance,
    y=X.columns,
    title="Feature Importance (Random Forest)",
    orientation="h"
)
st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------------------------------
# Prediction Form
# ----------------------------------------------------
st.header("Patient Lung Cancer Risk Prediction")

age = st.number_input("Age", 18, 100, 40)
smokes = st.selectbox("Smokes", ["No", "Yes"])
areaq = st.selectbox("AreaQ (Exposure)", ["No", "Yes"])
alkhol = st.selectbox("Alkhol", ["No", "Yes"])

smokes_val = 1 if smokes == "Yes" else 0
areaq_val = 1 if areaq == "Yes" else 0
alkhol_val = 1 if alkhol == "Yes" else 0

input_data = np.array([[age, smokes_val, areaq_val, alkhol_val]])

if st.button("Predict Risk"):
    scaled = scaler.transform(input_data)
    prob = log_model.predict_proba(scaled)[0][1]
    st.subheader(f"Predicted Lung Cancer Risk: {prob:.2%}")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.write("---")
st.write("Built by theolumayowa — Portfolio Project")
