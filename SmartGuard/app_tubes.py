import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

st.set_page_config(
    page_title="SmartGuard - Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ API Key Gemini tidak ditemukan di Secrets. Fitur Advisor nonaktif.")

@st.cache_resource
def load_gemini_model():
    if not GOOGLE_API_KEY:
        return None

    return genai.GenerativeModel("gemini-2.0-flash")

gemini_model = load_gemini_model()

def get_gemini_advice(suhu, rpm, torsi):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") 
        
        prompt = f"""
        Peran: Ahli Teknisi Senior.
        Kondisi Mesin: Suhu {suhu}K, Kecepatan {rpm}RPM, Torsi {torsi}Nm.
        Status: RISIKO KERUSAKAN TINGGI.
        Tugas: Berikan 3 instruksi perbaikan singkat dalam Bahasa Indonesia.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except:
            return f"Error: {e}"


@st.cache_resource
def load_smartguard_model():
    try:
        return joblib.load('smartguard_model.pkl')
    except Exception as e:
        st.error(f"File 'smartguard_model.pkl' tidak ditemukan! Error: {e}")
        return None

machine_model = load_smartguard_model()

st.title("🏭 SmartGuard: Industrial Predictive Maintenance")
st.write("Sistem prediksi kegagalan mesin berbasis AI untuk efisiensi operasional.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Input Parameter")
    
    type_prod = st.selectbox("Kualitas Produk (Type)", ['Low (L)', 'Medium (M)', 'High (H)'])
    type_map = {'Low (L)': 1, 'Medium (M)': 2, 'High (H)': 0}
    
    air_temp = st.number_input("Suhu Udara [K]", 290.0, 310.0, 300.0)
    proc_temp = st.number_input("Suhu Proses [K]", 300.0, 320.0, 310.0)
    rpm = st.number_input("Kecepatan Putar [RPM]", 1000, 3000, 1500)
    torque = st.number_input("Torsi [Nm]", 10.0, 100.0, 40.0)
    tool_wear = st.number_input("Keausan Alat [min]", 0, 300, 100)

    predict_btn = st.button("🔍 Cek Kondisi Mesin", use_container_width=True)

with col2:
    st.header("📊 Hasil Analisis")

    if predict_btn:
        if machine_model is not None:
            input_features = [[type_map[type_prod], air_temp, proc_temp, rpm, torque, tool_wear]]
            cols_name = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
            input_df = pd.DataFrame(input_features, columns=cols_name)

            prediction = machine_model.predict(input_df)[0]
            
            prob = None
            if hasattr(machine_model, "predict_proba"):
                prob = machine_model.predict_proba(input_df)[0][1]

            if prediction == 1:
                msg = f"🔴 BAHAYA! Mesin Terdeteksi AKAN RUSAK"
                if prob: msg += f" ({prob:.1%})"
                st.error(msg)

                st.divider()
                st.subheader("🤖 Saran Perbaikan (AI Advisor):")
                with st.spinner("Mengambil saran teknis..."):
                    saran = get_gemini_advice(proc_temp, rpm, torque)
                    st.info(saran)
            else:
                msg = "🟢 AMAN. Mesin Beroperasi Normal"
                if prob: msg += f" (Risiko: {prob:.1%})"
                st.success(msg)
                st.balloons()
        else:
            st.error("Model SmartGuard belum dimuat. Periksa file .pkl Anda.")
    else:
        st.info("Silakan masukkan parameter mesin dan tekan tombol prediksi.")