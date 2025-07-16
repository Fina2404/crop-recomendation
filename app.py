import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model dan Scaler
# -----------------------------
st.title("🌾 Crop Recommendation System")
st.caption("Gunakan kondisi tanah dan cuaca untuk mengetahui tanaman terbaik yang bisa ditanam.")

MODEL_PATH = "crop_model.keras"  
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# -----------------------------
# Petunjuk Pengisian (Tabel Markdown)
# -----------------------------
st.markdown("### 📌 Panduan Pengisian Parameter")
st.markdown("""
| Parameter         | Rentang Disarankan |
|-------------------|--------------------|
| Nitrogen (N)      | 0 – 129            |
| Fosfor (P)        | 6 – 143            |
| Kalium (K)        | 8 – 204            |
| Suhu (°C)         | 12 – 41            |
| Kelembaban (%)    | 15 – 97            |
| pH Tanah          | 4.6 – 8.7          |
| Curah Hujan (mm)  | 22 – 268           |
""")

# -----------------------------
# Input Form
# -----------------------------
st.header("Masukkan Parameter Tanah dan Cuaca")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 150.0, step=1.0, help="Rekomendasi: 0 – 129")
    temperature = st.number_input("Suhu (°C)", 0.0, 50.0, help="Rekomendasi: 12 – 41 °C")
    ph = st.number_input("pH Tanah", 0.0, 14.0, help="Rekomendasi: 4.6 – 8.7")

with col2:
    P = st.number_input("Fosfor (P)", 0.0, 150.0, step=1.0, help="Rekomendasi: 6 – 143")
    humidity = st.number_input("Kelembaban (%)", 0.0, 100.0, help="Rekomendasi: 15 – 97%")
    rainfall = st.number_input("Curah Hujan (mm)", 0.0, 300.0, help="Rekomendasi: 22 – 268 mm")

with col3:
    K = st.number_input("Kalium (K)", 0.0, 200.0, step=1.0, help="Rekomendasi: 8 – 204")

# -----------------------------
# Prediksi
# -----------------------------
label_mapping = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
    'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
    'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

if st.button("🔍 Prediksi Tanaman Terbaik"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_index = np.argmax(prediction)
    predicted_label = label_mapping[predicted_index]

    st.success(f"✅ Tanaman yang direkomendasikan adalah: **{predicted_label.capitalize()}**")

    st.write("\n")
    st.markdown("---")
    st.subheader("📊 Probabilitas Semua Tanaman:")
    prob_dict = {label_mapping[i]: float(prediction[0][i]) for i in range(len(label_mapping))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    st.bar_chart(sorted_probs)
