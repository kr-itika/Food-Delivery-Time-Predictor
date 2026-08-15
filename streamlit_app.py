import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)

@st.cache_resource
def load_resources():
    model = load_model("delivery_time_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_resources()

st.title("🚚 Food Delivery Time Predictor")
st.write("Enter the order details below to predict the estimated delivery time.")

st.subheader("Order Details")

col1, col2 = st.columns(2)

with col1:
    market_id = st.number_input("Market ID", min_value=0, value=1)

    store_category = st.text_input(
        "Store Category",
        value="american"
    )

    order_protocol = st.number_input(
        "Order Protocol",
        min_value=0,
        value=1
    )

    total_items = st.number_input(
        "Total Items",
        min_value=1,
        value=3
    )

    subtotal = st.number_input(
        "Subtotal",
        min_value=0.0,
        value=2500.0
    )

    num_distinct_items = st.number_input(
        "Number of Distinct Items",
        min_value=1,
        value=2
    )

    min_item_price = st.number_input(
        "Minimum Item Price",
        min_value=0.0,
        value=500.0
    )

    max_item_price = st.number_input(
        "Maximum Item Price",
        min_value=0.0,
        value=1500.0
    )

with col2:
    total_onshift_partners = st.number_input(
        "Total Onshift Partners",
        min_value=0,
        value=10
    )

    total_busy_partners = st.number_input(
        "Total Busy Partners",
        min_value=0,
        value=5
    )

    total_outstanding_orders = st.number_input(
        "Total Outstanding Orders",
        min_value=0,
        value=10
    )

    hour = st.slider(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.selectbox(
        "Day of Week (0 = Monday, 6 = Sunday)",
        range(7)
    )

    month = st.selectbox(
        "Month",
        range(1, 13)
    )

is_weekend = 1 if day_of_week >= 5 else 0

if st.button("Predict Delivery Time", type="primary"):

    input_data = pd.DataFrame({
        "market_id": [market_id],
        "store_primary_category": [store_category],
        "order_protocol": [order_protocol],
        "total_items": [total_items],
        "subtotal": [subtotal],
        "num_distinct_items": [num_distinct_items],
        "min_item_price": [min_item_price],
        "max_item_price": [max_item_price],
        "total_onshift_partners": [total_onshift_partners],
        "total_busy_partners": [total_busy_partners],
        "total_outstanding_orders": [total_outstanding_orders],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend]
    })

    processed_data = preprocessor.transform(input_data)

    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    prediction = model.predict(
        processed_data,
        verbose=0
    )[0][0]

    prediction = max(0, prediction)

    st.success(
        f"Estimated Delivery Time: {prediction:.2f} minutes"
    )

    st.info(
        f"Approximately {prediction / 60:.1f} hours."
    )