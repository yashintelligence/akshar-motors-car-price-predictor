import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Akshar Motors",
    page_icon="🚗",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #00D4FF;
}

.subtitle {
    font-size: 18px;
    color: #BBBBBB;
}

.prediction-box {
    background-color: #1E293B;
    padding: 25px;
    border-radius: 15px;
    text-align:center;
    border: 2px solid #00D4FF;
}

.prediction-price {
    font-size: 40px;
    font-weight:bold;
    color:#00FF88;
}

.footer {
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD FILES
# -------------------------

model = pickle.load(open("car_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -------------------------
# HEADER
# -------------------------

st.markdown(
    '<div class="title">🚗 AKSHAR MOTORS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Premium Pre-Owned Car Valuation System</div>',
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# INPUT SECTION
# -------------------------

st.subheader("📋 Vehicle Information")

col1, col2 = st.columns(2)

with col1:

    mileage = st.number_input(
        "Mileage (in thousand KM)",
        min_value=0.0,
        value=100.0
    )

    brand = st.selectbox(
        "Brand",
        [
            "BMW",
            "Mercedes-Benz",
            "Mitsubishi",
            "Renault",
            "Toyota",
            "Volkswagen"
        ]
    )

    body = st.selectbox(
        "Body Type",
        [
            "hatch",
            "other",
            "sedan",
            "vagon",
            "van"
        ]
    )

with col2:

    engineV = st.number_input(
        "Engine Volume (L)",
        min_value=0.0,
        value=2.0
    )

    engine_type = st.selectbox(
        "Engine Type",
        [
            "Gas",
            "Other",
            "Petrol"
        ]
    )

    registration = st.selectbox(
        "Registration Status",
        [
            "Yes",
            "No"
        ]
    )

st.divider()

# -------------------------
# BUTTON
# -------------------------

if st.button("💰 Calculate Market Value", use_container_width=True):

    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Numeric Features
    input_data.loc[0, "Mileage"] = mileage
    input_data.loc[0, "EngineV"] = engineV

    # Brand Encoding
    brand_col = f"Brand_{brand}"

    if brand_col in input_data.columns:
        input_data.loc[0, brand_col] = 1

    # Body Encoding
    body_col = f"Body_{body}"

    if body_col in input_data.columns:
        input_data.loc[0, body_col] = 1

    # Engine Encoding
    engine_col = f"Engine Type_{engine_type}"

    if engine_col in input_data.columns:
        input_data.loc[0, engine_col] = 1

    # Registration
    if registration == "Yes":
        input_data.loc[0, "Registration_yes"] = 1

    # Scale
    scaled_input = scaler.transform(input_data)

    # Predict
    log_prediction = model.predict(scaled_input)

    predicted_price = np.exp(log_prediction)

    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>Estimated Market Value</h3>
            <div class="prediction-price">
                ${predicted_price[0]:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Valuation Generated Successfully ✅")

    st.subheader("🔍 Vehicle Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Brand", brand)
    c2.metric("Mileage", mileage)
    c3.metric("Engine", engineV)

# -------------------------
# FOOTER
# -------------------------

st.divider()

st.markdown(
    """
    <div class="footer">
        Developed by Yash Patel • Akshar Motors Analytics Division
    </div>
    """,
    unsafe_allow_html=True
)