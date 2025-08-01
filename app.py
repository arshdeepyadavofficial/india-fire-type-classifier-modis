import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("C:/Users/arshdeep/Desktop/IMPORTANT/Deforestation/best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page title with custom styling
st.set_page_config(
    page_title=" Fire Type Classifier", 
    layout="centered",
    page_icon="🔥",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF4500, #FF6347, #FF8C00);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    .feature-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #FF4500, #FF6347);
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF4500, #FF6347) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(255, 69, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 69, 0, 0.6) !important;
    }
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 10px;
    }
    .stNumberInput > div > div {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App header with custom styling
st.markdown("""
<div class="main-header">
    <h1>🔥 Fire Type Classification</h1>
    <p>🛰️ AI-Powered MODIS Satellite Data Analysis</p>
</div>
""", unsafe_allow_html=True)

# Info section
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;'>
    <h3>🌍 Predict fire types using advanced machine learning and satellite readings</h3>
    <p>Enter the MODIS satellite parameters below to classify fire types with high accuracy</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌡️ Thermal Parameters")
    brightness = st.number_input("🔆 Brightness (K)", value=300.0, min_value=200.0, max_value=400.0, help="Surface temperature in Kelvin")
    bright_t31 = st.number_input("🌡️ Brightness T31 (K)", value=290.0, min_value=200.0, max_value=350.0, help="Brightness temperature at 11μm")
    frp = st.number_input("🔥 Fire Radiative Power (MW)", value=15.0, min_value=0.0, max_value=1000.0, help="Energy released by fire")

with col2:
    st.markdown("### 📊 Sensor Parameters")
    scan = st.number_input("📡 Scan Angle", value=1.0, min_value=0.0, max_value=60.0, help="Satellite scan angle")
    track = st.number_input("🛰️ Track Position", value=1.0, min_value=0.0, max_value=60.0, help="Satellite track position")
    confidence = st.selectbox("✅ Confidence Level", ["low", "nominal", "high"], 
                             help="Detection confidence level from satellite")

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Create feature container
st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
st.markdown("### 📈 Input Summary")

# Display input summary in a nice format
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔆 Brightness", f"{brightness}K")
    st.metric("🔥 FRP", f"{frp}MW")

with col2:
    st.metric("🌡️ Bright T31", f"{bright_t31}K")
    st.metric("📡 Scan", f"{scan}°")

with col3:
    st.metric("🛰️ Track", f"{track}°")
    st.metric("✅ Confidence", confidence.title())

st.markdown("</div>", unsafe_allow_html=True)

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Prediction section
st.markdown("---")
st.markdown("### 🎯 Fire Type Prediction")

# Center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Fire Type", use_container_width=True)

# Predict and display
if predict_button:
    with st.spinner("🔄 Analyzing satellite data..."):
        prediction = model.predict(scaled_input)[0]

        fire_types = {
            0: "🌿 Vegetation Fire",
            2: "🏭 Other Static Land Source",
            3: "🌊 Offshore Fire"
        }

        result = fire_types.get(prediction, "❓ Unknown")
        
        # Create beautiful prediction display
        if prediction == 0:
            color = "#228B22"  # Forest Green
            icon = "🌿"
            description = "Natural vegetation burning, typically forest or grassland fires"
        elif prediction == 2:
            color = "#FF8C00"  # Dark Orange
            icon = "🏭"
            description = "Industrial or urban heat sources, buildings, or infrastructure"
        elif prediction == 3:
            color = "#1E90FF"  # Dodger Blue
            icon = "🌊"
            description = "Marine or coastal fire sources, offshore platforms or vessels"
        else:
            color = "#808080"  # Gray
            icon = "❓"
            description = "Unknown fire type classification"

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22, {color}44); 
                    padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;
                    border-left: 5px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='color: {color}; margin: 0;'>{icon} Predicted Fire Type</h2>
            <h1 style='color: {color}; margin: 0.5rem 0; font-size: 2.5rem;'>{result}</h1>
            <p style='color: #666; font-size: 1.1rem; margin: 0;'>{description}</p>
            <div style='margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.7); 
                        border-radius: 10px; display: inline-block;'>
                <strong>Prediction Confidence: High ✅</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Add some spacing before footer
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Footer with copyright
st.markdown("""
<div class="footer">
    <p>🔥 Designed and Managed by <strong><a href="https://github.com/arshdeepyadavofficial" target="_blank" style="color: white; text-decoration: none;">Arshdeep Yadav</a></strong> | © 2025 Fire Classification System | 
    Powered by MODIS Satellite Data & Machine Learning 🛰️</p>
</div>
""", unsafe_allow_html=True)