import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="AgriIntel AI", layout="wide")

# ---------------------------------------------------
# Load Models
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

# ---------------------------------------------------
# Weather API
# ---------------------------------------------------

def fetch_weather(location):
    try:
        api_key = st.secrets["weather_api"]

        url = (
            f"http://api.weatherapi.com/v1/current.json"
            f"?key={api_key}&q={location}"
        )

        response = requests.get(url, timeout=10).json()

        if "current" not in response:
            st.error("Invalid location or Weather API issue.")
            return None

        temp = response["current"]["temp_c"]
        humidity = response["current"]["humidity"]
        rainfall = response["current"]["precip_mm"]

        return temp, humidity, rainfall

    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None


# ---------------------------------------------------
# Generic Soil Data
# ---------------------------------------------------

def get_default_soil_data():
    return {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 6.8,
    }


soil = get_default_soil_data()

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown(
    """
<style>

.stApp{
background:
linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
background-size: cover;
}

.main-title{
font-size:48px;
font-weight:800;
text-align:center;
color:#1b4332;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:20px;
margin-bottom:20px;
color:#444;
}

.weather-card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
margin-top:20px;
}

.weather-grid{
display:flex;
justify-content:space-around;
text-align:center;
}

.weather-box{
padding:10px;
}

.weather-value{
font-size:28px;
font-weight:bold;
color:#1b4332;
}

.weather-label{
font-size:14px;
color:gray;
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">AgriIntel AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Smart Crop and Fertilizer Recommendation System</div>',
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col2:

    st.subheader("Location")

    location = st.text_input(
        "Enter State, City, Town or Village",
        placeholder="Example: Delhi, Jaipur, Ludhiana, Pune, Meerut",
    )

with col1:

    st.subheader("Soil Nutrients")

    use_auto = st.checkbox("Use Default Soil Values", value=True)

    if use_auto:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0,
            max_value=200,
            value=soil["N"],
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0,
            max_value=200,
            value=soil["P"],
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0,
            max_value=200,
            value=soil["K"],
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=float(soil["ph"]),
        )

    else:

        N = st.number_input("Nitrogen (N)", 0, 200, 40)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 40)
        ph = st.number_input("Soil pH", 0.0, 14.0, 7.0)

predict = st.button("Predict Optimal Crop")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if predict:

    if not location.strip():
        st.warning("Please enter a valid location.")
        st.stop()

    weather = fetch_weather(location)

    if weather:

        temp, humidity, rainfall = weather

        # Weather Card
        st.markdown(
            f"""
            <div class="weather-card">

            <h3 style="text-align:center;color:#1b4332;">
            Live Weather Data
            </h3>

            <div class="weather-grid">

            <div class="weather-box">
            <div class="weather-value">{temp} °C</div>
            <div class="weather-label">Temperature</div>
            </div>

            <div class="weather-box">
            <div class="weather-value">{humidity} %</div>
            <div class="weather-label">Humidity</div>
            </div>

            <div class="weather-box">
            <div class="weather-value">{rainfall} mm</div>
            <div class="weather-label">Rainfall</div>
            </div>

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.warning("Using fallback weather values.")

        temp = 25
        humidity = 60
        rainfall = 100

    # ---------------------------------------------------
    # Crop Prediction
    # ---------------------------------------------------

    crop_input = np.array(
        [[N, P, K, temp, humidity, ph, rainfall]]
    )

    crop_probs = crop_model.predict_proba(crop_input)[0]

    top3_idx = np.argsort(crop_probs)[-3:][::-1]

    top3_crops = crop_encoder.inverse_transform(top3_idx)

    top3_conf = crop_probs[top3_idx]

    st.subheader("Top Crop Recommendations")

    for crop, conf in zip(top3_crops, top3_conf):

        st.progress(float(conf))

        st.write(
            f"**{crop} — {conf * 100:.2f}% confidence**"
        )

    # ---------------------------------------------------
    # Fertilizer Prediction
    # ---------------------------------------------------

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

    fert_probs = fert_model.predict_proba(fert_input)[0]

    fert_idx = np.argmax(fert_probs)

    fert_label = fert_encoder.inverse_transform(
        [fert_idx]
    )[0]

    fert_conf = fert_probs[fert_idx]

    st.subheader("Recommended Fertilizer")

    st.success(
        f"{fert_label} — {fert_conf * 100:.2f}% confidence"
    )

    # ---------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------

    st.subheader("Feature Importance")

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

        fig = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Importance",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Feature importance unavailable: {e}"
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "AgriIntel AI • Crop and Fertilizer Recommendation System • Powered by Machine Learning"
)
