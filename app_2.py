import streamlit as st
import numpy as np
import pickle

# 1. CONFIGURACION DE LA PAGINA
st.set_page_config(page_title="Clasificador de Vinos", layout="centered")
st.title("🍷 Clasificador de Variedad de Vino")
st.write("Ingresa las caracteristicas del vino para predecir su variedad.")

clases_vino = ["Variedad Altiplano", "Variedad Valle Central", "Variedad Costa"]


# 2. CARGAR EL PIPELINE ENTRENADO
@st.cache_resource
def cargar_modelo():
    nombre_archivo = "modelo_completo.pkl"
    try:
        with open(nombre_archivo, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error(f"❌ No se encontro el archivo **'{nombre_archivo}'**. Debe estar en la misma carpeta que app_2.py.")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo .pkl: {e}")
        return None


modelo_pipeline = cargar_modelo()

if modelo_pipeline is not None:
    # 3. FORMULARIO DE ENTRADA
    st.subheader("Caracteristicas del vino")

    alcohol = st.number_input("Alcohol", value=13.0, step=0.01, format="%.2f")
    malic_acid = st.number_input("Acido malico (malic_acid)", value=2.0, step=0.01, format="%.2f")
    ash = st.number_input("Ceniza (ash)", value=2.3, step=0.01, format="%.2f")
    alcalinity = st.number_input("Alcalinidad (alcalinity)", value=15.0, step=0.1, format="%.1f")

    # 4. PREDICCION
    if st.button("🚀 Predecir Variedad"):
        try:
            datos_entrada = np.array([[alcohol, malic_acid, ash, alcalinity]])
            id_clase = modelo_pipeline.predict(datos_entrada)[0]
            prediccion = clases_vino[id_clase]

            st.subheader("Resultado")
            st.success(f"🍇 Variedad predicha: **{prediccion}**")

        except Exception as e:
            st.error("❌ Ocurrio un error al calcular la prediccion. Verifica que los valores sean numericos validos.")
            st.write(f"**Detalle:** {e}")
