import streamlit as st
import pickle
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
import io

# -------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------
st.set_page_config(
    page_title="🐱🐶 Clasificador de Mascotas",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# ESTILOS PERSONALIZADOS
# -------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .cat-prediction {
        background-color: #FFE5B4;
        color: #FF8C00;
        border: 3px solid #FF8C00;
    }
    .dog-prediction {
        background-color: #E0F2F7;
        color: #1976D2;
        border: 3px solid #1976D2;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CARGAR MODELO
# -------------------------------------------------
@st.cache_resource
def cargar_modelo():
    try:
        with open("modelo_svm_mascotas_mejorado.pkl", "rb") as f:
            modelo_data = pickle.load(f)
        return modelo_data["model"], modelo_data["scaler"], modelo_data.get("config", {})
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo del modelo. Asegúrate de entrenar el modelo primero.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

# -------------------------------------------------
# PROCESAR IMAGEN
# -------------------------------------------------
def procesar_imagen(imagen_pil, img_height=128, img_width=128):
    """
    Procesa una imagen PIL y extrae características HOG
    """
    # Convertir PIL a numpy array
    img_array = np.array(imagen_pil)
    
    # Convertir a escala de grises
    if len(img_array.shape) == 3:
        img_gray = rgb2gray(img_array)
    else:
        img_gray = img_array
    
    # Redimensionar
    img_resized = resize(img_gray, (img_height, img_width), 
                        anti_aliasing=True, preserve_range=True)
    
    # Extraer características HOG
    features = hog(
        img_resized,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return features.reshape(1, -1)

# -------------------------------------------------
# FUNCIÓN DE PREDICCIÓN
# -------------------------------------------------
def predecir(modelo, scaler, features):
    """
    Realiza la predicción y retorna la clase y probabilidad
    """
    features_scaled = scaler.transform(features)
    prediccion = modelo.predict(features_scaled)[0]
    
    # Obtener probabilidades si el modelo lo soporta
    try:
        # Para SVM con probability=True
        probabilidades = modelo.predict_proba(features_scaled)[0]
        confianza = max(probabilidades) * 100
    except AttributeError:
        # Si no tiene predict_proba, usar decision_function
        decision = modelo.decision_function(features_scaled)[0]
        # Convertir a pseudo-probabilidad usando sigmoid
        confianza = 1 / (1 + np.exp(-abs(decision))) * 100
    
    clase = "Gato 🐱" if prediccion == 0 else "Perro 🐶"
    return clase, confianza

# -------------------------------------------------
# INTERFAZ PRINCIPAL
# -------------------------------------------------
def main():
    # Encabezado
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🐾 Clasificador de Gatos y Perros</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7F8C8D; font-size: 18px;'>Sube una imagen y descubre si es un gato o un perro</p>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cargar modelo
    modelo, scaler, config = cargar_modelo()
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Modelo")
        
        if config:
            st.metric("Precisión en Entrenamiento", f"{config.get('train_accuracy', 0)*100:.1f}%")
            st.metric("Precisión en Prueba", f"{config.get('test_accuracy', 0)*100:.1f}%")
        
        st.markdown("---")
        st.subheader("📋 Instrucciones")
        st.write("""
        1. Sube una imagen de un gato o perro
        2. Espera el procesamiento
        3. ¡Obtén tu predicción!
        
        **Formatos aceptados:** JPG, JPEG, PNG
        """)
        
        st.markdown("---")
        st.subheader("💡 Consejos")
        st.write("""
        - Usa imágenes claras y bien iluminadas
        - Asegúrate de que el animal sea visible
        - Mejores resultados con un solo animal
        """)
    
    # Área de carga de imagen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png"],
            help="Sube una imagen de un gato o perro"
        )
    
    # Procesamiento y predicción
    if uploaded_file is not None:
        # Cargar y mostrar imagen
        imagen = Image.open(uploaded_file)
        
        # Convertir a RGB si es necesario
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Mostrar imagen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(imagen, caption="Imagen cargada", use_container_width=True)
        
        # Botón de predicción
        if st.button("🔍 Clasificar Imagen"):
            with st.spinner("🤖 Analizando la imagen..."):
                try:
                    # Procesar imagen
                    img_height = config.get('img_height', 128)
                    img_width = config.get('img_width', 128)
                    features = procesar_imagen(imagen, img_height, img_width)
                    
                    # Realizar predicción
                    clase, confianza = predecir(modelo, scaler, features)
                    
                    # Mostrar resultado
                    st.success("✅ ¡Clasificación completada!")
                    
                    # Mostrar predicción con estilo
                    if "Gato" in clase:
                        st.markdown(f"""
                        <div class="prediction-box cat-prediction">
                            🐱 ¡Es un GATO! 🐱
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box dog-prediction">
                            🐶 ¡Es un PERRO! 🐶
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mostrar confianza
                    st.markdown(f"""
                    <div style='text-align: center; font-size: 20px; color: #34495E; margin-top: 10px;'>
                        <b>Confianza:</b> {confianza:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barra de progreso visual
                    st.progress(confianza / 100)
                    
                    # Interpretación de confianza
                    if confianza > 80:
                        st.info("💯 Alta confianza en la predicción")
                    elif confianza > 60:
                        st.warning("⚠️ Confianza moderada - La imagen podría ser ambigua")
                    else:
                        st.error("🤔 Baja confianza - Intenta con una imagen más clara")
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen: {e}")
        
        # Botón para limpiar
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clasificar otra imagen"):
            st.rerun()
    
    else:
        # Mensaje cuando no hay imagen
        st.info("👆 Sube una imagen para comenzar")
        
        # Ejemplos visuales
        st.markdown("---")
        st.subheader("📸 Ejemplos de uso")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #FFE5B4; border-radius: 10px;'>
                <h3>🐱 Gatos</h3>
                <p>Imágenes claras de gatos en diferentes poses y razas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background-color: #E0F2F7; border-radius: 10px;'>
                <h3>🐶 Perros</h3>
                <p>Imágenes claras de perros en diferentes poses y razas</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #95A5A6; padding: 20px;'>
        <p>Desarrollado con ❤️ usando Machine Learning y Streamlit</p>
        <p><small>Modelo: SVM con características HOG</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()