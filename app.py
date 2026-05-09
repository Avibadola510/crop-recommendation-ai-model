import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AgriIntel AI",
    page_icon="🌱",
    layout="wide",
)

# =========================================================
# LOAD MODELS
# =========================================================

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
        metadata,
    )


(
    crop_model,
    crop_encoder,
    fert_model,
    fert_encoder,
    fert_crop_encoder,
    metadata,
) = load_models()

# =========================================================
# WEATHER API
# =========================================================

def fetch_weather(location):

    try:

        api_key = st.secrets["weather_api"]

        url = (
            f"http://api.weatherapi.com/v1/forecast.json"
            f"?key={api_key}&q={location}&days=7"
        )

        response = requests.get(url, timeout=10).json()

        if "current" not in response:
            return None

        current = response["current"]

        temp = current["temp_c"]
        humidity = current["humidity"]
        rainfall = current["precip_mm"]

        forecast_days = response["forecast"]["forecastday"]

        forecast_data = []

        for day in forecast_days:

            forecast_data.append(
                {
                    "date": day["date"],
                    "temp": day["day"]["avgtemp_c"],
                    "rain": day["day"]["totalprecip_mm"],
                }
            )

        return temp, humidity, rainfall, forecast_data

    except Exception as e:

        st.error(f"Weather API Error: {e}")

        return None


# =========================================================
# DEFAULT SOIL DATA
# =========================================================

def get_default_soil():

    return {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 6.8,
    }


soil = get_default_soil()

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp{
    background:
    linear-gradient(rgba(8,12,18,0.90), rgba(8,12,18,0.92)),
    url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
    background-size: cover;
    background-position: center;
    color: white;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background: rgba(16,24,32,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Title */

.main-title{
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.subtitle{
    font-size: 18px;
    color: #b8c2cc;
    margin-bottom: 30px;
}

/* Cards */

.glass-card{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* Metric Cards */

.metric-card{
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}

.metric-value{
    font-size: 34px;
    font-weight: 700;
    color: #6ee7b7;
}

.metric-label{
    color: #cbd5e1;
    margin-top: 8px;
}

/* Crop Cards */

.crop-card{
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */

.stButton>button{
    width: 100%;
    background: linear-gradient(135deg,#22c55e,#15803d);
    color: white;
    border: none;
    padding: 16px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover{
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(34,197,94,0.35);
}

/* Inputs */

.stTextInput>div>div>input{
    background: rgba(255,255,255,0.08);
    color: white;
    border-radius: 12px;
}

.stNumberInput input{
    background: rgba(255,255,255,0.08);
    color: white;
}

/* Tabs */

.stTabs [data-baseweb="tab"]{
    font-size: 16px;
}

/* Footer */

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
    padding:20px;
}

/* Hide Streamlit Branding */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## AgriIntel AI")

    st.markdown("---")

    location = st.text_input(
        "Location",
        placeholder="Enter city, district, village..."
    )

    st.markdown("### Soil Nutrients")

    use_default = st.checkbox(
        "Use Default Soil Values",
        value=True
    )

    if use_default:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0,
            max_value=200,
            value=soil["N"]
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0,
            max_value=200,
            value=soil["P"]
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0,
            max_value=200,
            value=soil["K"]
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=float(soil["ph"])
        )

    else:

        N = st.number_input("Nitrogen (N)", 0, 200, 40)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 40)
        ph = st.number_input("Soil pH", 0.0, 14.0, 7.0)

    predict = st.button("Predict Crop")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="glass-card">

<div class="main-title">
AgriIntel AI
</div>

<div class="subtitle">
AI Powered Crop and Fertilizer Recommendation System
</div>

</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# MAIN PREDICTION
# =========================================================

if predict:

    if not location.strip():

        st.warning("Please enter a valid location.")
        st.stop()

    weather = fetch_weather(location)

    if weather:

        temp, humidity, rainfall, forecast_data = weather

    else:

        temp = 25
        humidity = 60
        rainfall = 100
        forecast_data = []

    # =====================================================
    # WEATHER METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{temp}°C</div>
                <div class="metric-label">Temperature</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{humidity}%</div>
                <div class="metric-label">Humidity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{rainfall} mm</div>
                <div class="metric-label">Rainfall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # WEATHER ALERTS
    # =====================================================

    if temp > 40:
        st.error("Extreme heat detected. Crops may experience heat stress.")

    elif rainfall < 10:
        st.warning("Low rainfall detected. Irrigation may be required.")

    else:
        st.success("Current weather conditions are stable for farming.")

    # =====================================================
    # PREDICTION
    # =====================================================

    crop_input = np.array(
        [[N, P, K, temp, humidity, ph, rainfall]]
    )

    crop_probs = crop_model.predict_proba(crop_input)[0]

    top3_idx = np.argsort(crop_probs)[-3:][::-1]

    top3_crops = crop_encoder.inverse_transform(top3_idx)

    top3_conf = crop_probs[top3_idx]

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "Crop Recommendations",
            "Weather Analytics",
            "AI Insights",
        ]
    )

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.markdown("## Recommended Crops")

        crop_info = {
            "rice": {
                "season": "Kharif",
                "water": "High",
                "duration": "120 Days",
            },
            "wheat": {
                "season": "Rabi",
                "water": "Moderate",
                "duration": "110 Days",
            },
            "maize": {
                "season": "Kharif",
                "water": "Moderate",
                "duration": "90 Days",
            },
        }

        for crop, conf in zip(top3_crops, top3_conf):

            crop_lower = crop.lower()

            info = crop_info.get(
                crop_lower,
                {
                    "season": "Mixed",
                    "water": "Moderate",
                    "duration": "100 Days",
                },
            )

            st.markdown(
                f"""
                <div class="crop-card">

                <h2 style="color:#6ee7b7;">
                {crop}
                </h2>

                <p style="font-size:18px;">
                Confidence Score: <b>{conf*100:.2f}%</b>
                </p>

                <p>
                Season: {info['season']}
                </p>

                <p>
                Water Requirement: {info['water']}
                </p>

                <p>
                Crop Duration: {info['duration']}
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(float(conf))

        # Fertilizer

        crop_encoded = fert_crop_encoder.transform(
            [top3_crops[0]]
        )[0]

        fert_input = np.array(
            [[
                N,
                P,
                K,
                temp,
                humidity,
                rainfall,
                crop_encoded
            ]]
        )

        fert_probs = fert_model.predict_proba(
            fert_input
        )[0]

        fert_idx = np.argmax(fert_probs)

        fert_label = fert_encoder.inverse_transform(
            [fert_idx]
        )[0]

        fert_conf = fert_probs[fert_idx]

        st.markdown("## Recommended Fertilizer")

        st.markdown(
            f"""
            <div class="glass-card">

            <h2 style="color:#6ee7b7;">
            {fert_label}
            </h2>

            <p style="font-size:18px;">
            Recommendation Confidence:
            <b>{fert_conf*100:.2f}%</b>
            </p>

            <p>
            Suitable for current soil and weather conditions.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.markdown("## 7-Day Weather Forecast")

        if forecast_data:

            forecast_df = pd.DataFrame(forecast_data)

            fig = px.line(
                forecast_df,
                x="date",
                y="temp",
                markers=True,
                title="Temperature Forecast",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            fig2 = px.bar(
                forecast_df,
                x="date",
                y="rain",
                title="Rainfall Forecast",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.markdown("## AI Powered Insights")

        # Soil Analysis

        if ph < 6:
            st.warning(
                "Soil is acidic. Lime treatment may improve productivity."
            )

        elif ph > 8:
            st.warning(
                "Soil is alkaline. Gypsum treatment may help."
            )

        else:
            st.success(
                "Soil pH is within optimal range."
            )

        if N < 30:
            st.error(
                "Nitrogen levels are low."
            )

        elif N > 100:
            st.warning(
                "Nitrogen levels are excessive."
            )

        else:
            st.success(
                "Nitrogen levels are balanced."
            )

        # Feature Importance

        st.markdown("## Feature Importance")

        try:

            importances = (
                crop_model.calibrated_classifiers_[0]
                .estimator.feature_importances_
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": metadata["crop_features"],
                    "Importance": importances,
                }
            ).sort_values(
                by="Importance",
                ascending=False
            )

            fig3 = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                color="Importance",
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Feature importance unavailable: {e}"
            )

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = f"""
AgriIntel AI Report
Generated: {datetime.now()}

Location: {location}

Temperature: {temp}
Humidity: {humidity}
Rainfall: {rainfall}

Top Crop: {top3_crops[0]}

Recommended Fertilizer:
{fert_label}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="agriintel_report.txt",
        mime="text/plain",
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">

AgriIntel AI

AI Powered Smart Agriculture Platform

</div>
""",
    unsafe_allow_html=True,
)
