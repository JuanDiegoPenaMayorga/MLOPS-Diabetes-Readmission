import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Predicción de Readmisión de Diabetes",
    page_icon="🩺",
    layout="centered"
)

API_URL = "http://diabetes-api-service/predict"
st.title("Predicción de Readmisión por Diabetes")
st.markdown("Aplicación ML para predecir readmision en menos de 30 días.")

with st.form("prediction_form"):
    st.header("Información del Paciente (Valores de ejemplo)")
    col1, col2 = st.columns(2)

    with col1:
        race = st.selectbox("Raza (Race)", ["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian"])
        gender = st.selectbox("Género (Gender)", ["Female", "Male"])
        age = st.selectbox("Rango de Edad (Age)", ['[70-80)', '[60-70)', '[50-60)', '[80-90)', '[40-50)'])
        time_in_hospital = st.number_input("Días en Hospital", min_value=1, value=5)
        num_lab_procedures = st.number_input("Nº Procedimientos de Laboratorio", min_value=1, value=40)

    with col2:
        num_procedures = st.number_input("Nº Procedimientos Médicos", min_value=0, value=1)
        num_medications = st.number_input("Nº de Medicamentos", min_value=1, value=15)
        diag_1 = st.text_input("Diagnóstico Principal", value="250.83")
        diag_2 = st.text_input("Diagnóstico Secundario", value="401")
        diag_3 = st.text_input("Diagnóstico Adicional", value="250.01")
    
    submit_button = st.form_submit_button(label="Realizar Predicción")

if submit_button:
    payload = {
        "race": race, "gender": gender, "age": age,
        "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures, "num_medications": num_medications,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
    }
    try:
        with st.spinner("Enviando datos al modelo..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            probability = float(result.get("probability", 0))

            st.success("¡Predicción recibida!")
            if prediction == "YES":
                st.warning(f"**Resultado: SÍ será readmitido.**")
            else:
                st.info(f"**Resultado: NO será readmitido.**")
            
            st.metric(label="Confianza de la Predicción", value=f"{probability:.2%}")
        else:
            st.error(f"Error de la API (Código {response.status_code}): {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: No se pudo conectar a la API. Detalle: {e}")
