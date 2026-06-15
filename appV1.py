import streamlit as st
import pandas as pd
import joblib
import os

# Set up page configuration
st.set_page_config(
    page_title="NeoBank Fraud Detector",
    page_icon="🛡️",
    layout="centered"
)

# Title and Story Context
st.title("🛡️ Credit Card Fraud Detection Sandbox")
st.caption("A Streamlit companion app for the NeoBank ML Workshop")

st.info(
    "👉 **The Scenario:** A new transaction has just hit NeoBank's systems. "
    "Use the controls below to tweak the transaction parameters and see if our "
    "trained Random Forest model flags it as a threat!"
)

# --- LOAD TRAINED MODEL ---
MODEL_PATH = 'fraud_detector_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

# Handle missing model file gracefully
if model is None:
    st.error(f"⚠️ **Model file not found!** Please make sure `{MODEL_PATH}` is in the same directory as this script.")
    st.stop()

# --- INTERACTIVE USER INPUTS ---
st.subheader("📋 Transaction Details")

# Organize inputs into columns for a clean UI
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=450.0, step=10.0)
    hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=3)
    distance = st.number_input("Distance from Home (km)", min_value=0.0, max_value=5000.0, value=180.0, step=5.0)

with col2:
    velocity = st.number_input("Transactions in Last Hour", min_value=0, max_value=50, value=5)
    is_online = st.toggle("Is Online Transaction?", value=True)
    is_foreign = st.toggle("Is Foreign Transaction?", value=True)

# Dropdown for Merchant Category
category = st.selectbox(
    "Merchant Category",
    options=["Electronics", "Gas Station", "Grocery", "Online Shopping", "Travel", "Other (Default)"]
)

# --- PREPROCESSING & MAPPING ---
# Initialize all One-Hot Encoded category columns to 0
cat_mapping = {
    'merchant_category_electronics': 0,
    'merchant_category_gas_station': 0,
    'merchant_category_grocery': 0,
    'merchant_category_online_shopping': 0,
    'merchant_category_travel': 0
}

# Turn on the specific column based on selection
if category == "Electronics":
    cat_mapping['merchant_category_electronics'] = 1
elif category == "Gas Station":
    cat_mapping['merchant_category_gas_station'] = 1
elif category == "Grocery":
    cat_mapping['merchant_category_grocery'] = 1
elif category == "Online Shopping":
    cat_mapping['merchant_category_online_shopping'] = 1
elif category == "Travel":
    cat_mapping['merchant_category_travel'] = 1

# Build the mock transaction payload exactly how the model expects it
mock_tx = pd.DataFrame([{
    'amount': amount,
    'hour': hour,
    'distance_from_home': distance,
    'is_online': 1 if is_online else 0,
    'is_foreign': 1 if is_foreign else 0,
    'velocity_last_hour': velocity,
    **cat_mapping
}])

# --- PREDICTION ENGINE ---
st.markdown("---")

if st.button("🔍 Run Fraud Risk Analysis", type="primary", use_container_width=True):
    # Predict using the loaded model
    prediction = model.predict(mock_tx)[0]
    probability = model.predict_proba(mock_tx)[0][1]
    
    # Display Results
    st.subheader("📊 Analysis Verdict")
    
    if prediction == 1:
        st.error(f"🚨 **[ALERT] FRAUDULENT**")
        st.metric(label="Fraud Confidence Probability", value=f"{probability * 100:.1f}%")
        st.warning("Action Required: This transaction triggers high-risk parameters. Freeze card and notify client.")
    else:
        st.success(f"✅ **[OK] LEGITIMATE**")
        st.metric(label="Fraud Confidence Probability", value=f"{probability * 100:.1f}%")
        st.info("Action Required: Transaction looks clean. Approved for processing.")
