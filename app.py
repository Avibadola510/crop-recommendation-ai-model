import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AgriIntel AI",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():

    crop_model = joblib.load("models/crop_model.pkl")
    crop_encoder = joblib.load("models/crop_encoder.pkl")

    fert_model = joblib.load("models/fertilizer_model.pkl")
    fert_encoder = joblib.load("models/fertilizer_encoder.pkl")
    fert_crop_encoder = joblib.load("models/fert_crop_encoder.pkl")

    metadata = joblib.load("models/metadata.pkl")

    return (
        crop_model,
        crop_encoder,
        fert_model,
        fert_encoder,
        fert_crop_encoder,
        metadata
    )


(
    crop_model,
    crop_encoder,
    fert_model,
    fert_encoder,
    fert_crop_encoder,
    metadata
) = load_models()

# ---------------------------------------------------
# WEATHER API
# ---------------------------------------------------
def fetch_weather(city):

    try:
        api_key = st.secrets["weather_api"]

        url = (
            f"http://api.weatherapi.com/v1/current.json?"
            f"key={api_key}&q={city}"
        )

        response = requests.get(url).json()

        temp = response["current"]["temp_c"]
        humidity = response["current"]["humidity"]
        rainfall = response["current"]["precip_mm"]

        return temp, humidity, rainfall

    except:
        return None


# ---------------------------------------------------
# SMART SOIL ESTIMATION
# ---------------------------------------------------
def get_soil_data(temp, humidity, rainfall):

    if rainfall > 200:

        N = 90
        P = 45
        K = 45
        ph = 6.5

    elif rainfall > 100:

        N = 70
        P = 40
        K = 40
        ph = 6.8

    else:

        N = 50
        P = 35
        K = 35
        ph = 7.2

    return {
        "N": N,
        "P": P,
        "K": K,
        "ph": ph
    }


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.stApp{
background-color:#f5f7fa;
font-family:'Segoe UI', sans-serif;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:4rem;
padding-right:4rem;
}

.main-title{
font-size:42px;
font-weight:700;
text-align:center;
color:#111827;
margin-bottom:10px;
}

.sub-title{
text-align:center;
font-size:18px;
color:#6b7280;
margin-bottom:40px;
}

.card{
background:white;
padding:25px;
border-radius:14px;
border:1px solid #e5e7eb;
box-shadow:0 4px 10px rgba(0,0,0,0.04);
margin-bottom:20px;
}

.weather-grid{
display:flex;
justify-content:space-around;
text-align:center;
margin-top:20px;
}

.weather-box{
padding:10px;
}

.weather-value{
font-size:28px;
font-weight:700;
color:#111827;
}

.weather-label{
font-size:14px;
color:#6b7280;
}

.stButton > button{
width:100%;
background:#111827;
color:white;
padding:14px;
font-size:16px;
font-weight:600;
border:none;
border-radius:10px;
transition:0.3s;
}

.stButton > button:hover{
background:#374151;
}

input{
border-radius:8px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="main-title">AgriIntel AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Crop and Fertilizer Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("Location")

    location = st.text_input(
        "Enter City or District",
        placeholder="Delhi, Jaipur, Pune, Lucknow"
    )

with col2:

    st.subheader("Prediction Settings")

    use_manual = st.checkbox(
        "Manually Edit Soil Values",
        value=False
    )

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
predict = st.button("Predict Crop")

# ---------------------------------------------------
# MAIN PREDICTION
# ---------------------------------------------------
if predict:

    if not location:

        st.warning("Please enter a valid location.")

    else:

        with st.spinner("Fetching live weather data..."):

            weather = fetch_weather(location)

        if weather:

            temp, humidity, rainfall = weather

            soil = get_soil_data(
                temp,
                humidity,
                rainfall
            )

            # ---------------------------------------------------
            # SOIL VALUES
            # ---------------------------------------------------
            if use_manual:

                st.subheader("Soil Parameters")

                col3, col4 = st.columns(2)

                with col3:

                    N = st.number_input(
                        "Nitrogen",
                        0,
                        200,
                        soil["N"]
                    )

                    P = st.number_input(
                        "Phosphorus",
                        0,
                        200,
                        soil["P"]
                    )

                with col4:

                    K = st.number_input(
                        "Potassium",
                        0,
                        200,
                        soil["K"]
                    )

                    ph = st.number_input(
                        "Soil pH",
                        0.0,
                        14.0,
                        soil["ph"]
                    )

            else:

                N = soil["N"]
                P = soil["P"]
                K = soil["K"]
                ph = soil["ph"]

            # ---------------------------------------------------
            # WEATHER CARD
            # ---------------------------------------------------
            st.markdown(f"""
            <div class="card">

            <h3 style="text-align:center;">
            Live Environmental Data
            </h3>

            <div class="weather-grid">

                <div class="weather-box">
                    <div class="weather-value">
                    {temp}°C
                    </div>
                    <div class="weather-label">
                    Temperature
                    </div>
                </div>

                <div class="weather-box">
                    <div class="weather-value">
                    {humidity}%
                    </div>
                    <div class="weather-label">
                    Humidity
                    </div>
                </div>

                <div class="weather-box">
                    <div class="weather-value">
                    {rainfall} mm
                    </div>
                    <div class="weather-label">
                    Rainfall
                    </div>
                </div>

            </div>

            </div>
            """, unsafe_allow_html=True)

            # ---------------------------------------------------
            # CROP PREDICTION
            # ---------------------------------------------------
            crop_input = np.array([[
                N,
                P,
                K,
                temp,
                humidity,
                ph,
                rainfall
            ]])

            crop_probs = crop_model.predict_proba(crop_input)[0]

            top3_idx = np.argsort(crop_probs)[-3:][::-1]

            top3_crops = crop_encoder.inverse_transform(top3_idx)

            top3_conf = crop_probs[top3_idx]

            st.subheader("Recommended Crops")

            for crop, conf in zip(top3_crops, top3_conf):

                st.write(
                    f"{crop} — {conf * 100:.2f}% confidence"
                )

                st.progress(float(conf))

            # ---------------------------------------------------
            # FERTILIZER PREDICTION
            # ---------------------------------------------------
            crop_encoded = fert_crop_encoder.transform(
                [top3_crops[0]]
            )[0]

            fert_input = np.array([[
                N,
                P,
                K,
                temp,
                humidity,
                rainfall,
                crop_encoded
            ]])

            fert_probs = fert_model.predict_proba(
                fert_input
            )[0]

            fert_idx = np.argmax(fert_probs)

            fert_label = fert_encoder.inverse_transform(
                [fert_idx]
            )[0]

            fert_conf = fert_probs[fert_idx]

            st.subheader("Recommended Fertilizer")

            st.success(
                f"{fert_label} — "
                f"{fert_conf * 100:.2f}% confidence"
            )

            # ---------------------------------------------------
            # FEATURE IMPORTANCE
            # ---------------------------------------------------
            st.subheader("Feature Importance")

            importances = (
                crop_model
                .calibrated_classifiers_[0]
                .estimator
                .feature_importances_
            )

            importance_df = pd.DataFrame({
                "Feature": metadata["crop_features"],
                "Importance": importances
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            fig = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                color="Importance"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.error(
                "Unable to fetch weather data for this location."
            )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.caption(
    "AgriIntel AI • Machine Learning Based "
    "Agricultural Recommendation System"
)
