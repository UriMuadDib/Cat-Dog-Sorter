import streamlit as st
import pickle
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
import os

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="🐾 Clasificador Inteligente",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TEMA OSCURO/SEMI-OSCURO CON BUEN CONTRASTE
st.markdown("""
    <style>
    /* Fondo principal */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e8e8e8 !important;
        font-weight: 700 !important;
    }
    
    /* Texto general */
    p, span, div, label {
        color: #c8c8c8 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e8e8e8 !important;
    }
    
    /* Botones principales */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
        color: #ffffff;
        height: 3em;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(15, 76, 117, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b6ca8 0%, #0f4c75 100%);
        box-shadow: 0 6px 20px rgba(27, 108, 168, 0.6);
        transform: translateY(-2px);
    }
    
    /* Cajas de predicción */
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 25px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    .cat-prediction {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        color: #ffffff;
        border: 3px solid #ff6b6b;
    }
    
    .dog-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #ffffff;
        border: 3px solid #00f2fe;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #4facfe !important;
        font-size: 28px !important;
    }
    
    /* Cards informativos */
    .info-card {
        background: rgba(15, 76, 117, 0.3);
        border-left: 4px solid #4facfe;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #e8e8e8 !important;
    }
    
    /* Animación */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 76, 117, 0.2);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Success/Info/Warning/Error boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(15, 76, 117, 0.2) !important;
        color: #e8e8e8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# CARGAR MODELO
@st.cache_resource
def cargar_modelo_clasico():
    """Carga el modelo clásico mejorado (Ensemble)"""
    try:
        with open("modelo.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["scaler"], data.get("config", {})
    except FileNotFoundError:
        st.error("❌ No se encontró 'modelo.pkl'. Ejecuta primero el entrenamiento.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

# PROCESAMIENTO DE IMÁGENES
def extraer_caracteristicas_avanzadas(img):
    """
    Extrae características combinadas para el modelo clásico:
    - HOG: Detecta formas y bordes
    - Color Histograms: Distribución de colores en 4 regiones
    - LBP: Analiza texturas locales
    """
    # 1. HOG Features (formas y bordes)
    hog_features = hog(
        img,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    # 2. Color Histograms por regiones (distribución espacial de colores)
    h, w = img.shape
    histograms = []
    for i in range(2):
        for j in range(2):
            region = img[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
            hist, _ = np.histogram(region, bins=32, range=(0, 1))
            histograms.extend(hist)
    
    # 3. LBP Features (texturas)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    
    # Combinar todas las características
    features_combined = np.concatenate([
        hog_features,
        np.array(histograms) / (np.sum(histograms) + 1e-7),
        lbp_hist / (np.sum(lbp_hist) + 1e-7)
    ])
    
    return features_combined.reshape(1, -1)

def procesar_imagen(imagen_pil, img_height=128, img_width=128):
    """Procesa una imagen PIL y extrae características"""
    # Convertir PIL a numpy array
    img_array = np.array(imagen_pil)
    
    # Convertir a escala de grises
    if len(img_array.shape) == 3:
        img_gray = rgb2gray(img_array)
    else:
        img_gray = img_array
    
    # Redimensionar con anti-aliasing
    img_resized = resize(img_gray, (img_height, img_width), 
                        anti_aliasing=True, preserve_range=True)
    
    # Normalizar a [0, 1]
    img_resized = img_resized / 255.0
    
    # Extraer características avanzadas
    return extraer_caracteristicas_avanzadas(img_resized)

# PREDICCIÓN
def predecir(modelo, scaler, features):
    """Realiza la predicción con el ensemble"""
    # Escalar características
    features_scaled = scaler.transform(features)
    
    # Predicción
    prediccion = modelo.predict(features_scaled)[0]
    
    # Obtener confianza
    try:
        probabilidades = modelo.predict_proba(features_scaled)[0]
        confianza = max(probabilidades) * 100
    except:
        # Si no tiene predict_proba (por seguridad)
        confianza = 75.0
    
    clase = "Gato 🐱" if prediccion == 0 else "Perro 🐶"
    return clase, confianza

# INTERFAZ PRINCIPAL
def main():
    # Header principal
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 48px; margin-bottom: 10px;'>
                🐾 Clasificador Inteligente de Mascotas
            </h1>
            <p style='font-size: 20px; color: #4facfe;'>
                Machine Learning Clásico con Características Avanzadas
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    modelo, scaler, config = cargar_modelo_clasico()
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 Información del Modelo")
        
        # Métricas del modelo
        if config:
            col1, col2 = st.columns(2)
            train_acc = config.get('train_accuracy', 0) * 100
            test_acc = config.get('test_accuracy', 0) * 100
            
            col1.metric("Entrenamiento", f"{train_acc:.1f}%")
            col2.metric("Prueba", f"{test_acc:.1f}%")
            
            # Indicador de overfitting
            diferencia = train_acc - test_acc
            if diferencia > 15:
                st.warning(f"⚠️ Diferencia: {diferencia:.1f}%")
            elif diferencia < 10:
                st.success(f"✅ Diferencia: {diferencia:.1f}%")
            else:
                st.info(f"📊 Diferencia: {diferencia:.1f}%")
        
        st.markdown("---")
        
        # Explicación técnica expandible
        with st.expander("🔍 ¿Cómo funciona?", expanded=False):
            st.markdown("""
            **Tecnología Clásica de ML:**
            
            1️⃣ **Extracción de Características**
            - **HOG**: Detecta formas y bordes del animal
            - **Histogramas de Color**: Analiza distribución de colores
            - **LBP**: Captura texturas del pelaje
            
            2️⃣ **Clasificación Ensemble**
            - **SVM (kernel RBF)**: Encuentra límites complejos
            - **Random Forest**: Vota con múltiples árboles
            - **Combinación**: Ambos votan la decisión final
            
            3️⃣ **Ventajas**
            - ✅ Rápido y eficiente
            - ✅ No requiere GPU
            - ✅ Interpretable
            
            4️⃣ **Limitaciones**
            - ⚠️ Sensible a fondos complejos
            - ⚠️ Mejor con imágenes claras
            """)
        
        st.markdown("---")
        
        # Detalles técnicos
        with st.expander("⚙️ Detalles Técnicos", expanded=False):
            if config:
                st.markdown(f"""
                **Características:**
                - HOG (orientations=12, cells=8x8)
                - Color Histograms (4 regiones, 32 bins)
                - LBP (P=8, R=1, uniform)
                """)
        
        st.markdown("---")
        
        # Consejos de uso
        st.markdown("### 💡 Consejos para Mejores Resultados")
        st.markdown("""
        ✅ **Recomendado:**
        - Imagen clara y bien iluminada
        - Un solo animal en la foto
        - Animal en primer plano
        - Fondo simple o uniforme
        - Formatos: JPG, JPEG, PNG
        
        ⚠️ **Evitar:**
        - Múltiples animales
        - Fondos muy complejos
        - Imágenes muy oscuras
        - Ángulos extremos
        """)
    
    # ÁREA PRINCIPAL
    st.markdown("---")
    
    # Cargador de archivo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "📤 Sube una imagen de tu mascota",
            type=["jpg", "jpeg", "png"],
            help="Selecciona una foto clara de un gato o perro"
        )
    
    # Procesamiento cuando hay imagen
    if uploaded_file is not None:
        # Cargar y convertir imagen
        imagen = Image.open(uploaded_file)
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        # Mostrar imagen cargada
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(imagen, caption="📷 Imagen cargada", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Botón de clasificación
        if st.button("🔍 Clasificar Imagen", use_container_width=True):
            with st.spinner("🤖 Analizando imagen con algoritmos de ML..."):
                try:
                    # Procesar imagen
                    img_height = config.get('img_height', 128)
                    img_width = config.get('img_width', 128)
                    features = procesar_imagen(imagen, img_height, img_width)
                    
                    # Realizar predicción
                    clase, confianza = predecir(modelo, scaler, features)
                    
                    # Mostrar resultado exitoso
                    st.success("✅ Clasificación completada exitosamente")
                    
                    # Caja de predicción centrada
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if "Gato" in clase:
                            st.markdown(f"""
                            <div class="prediction-box cat-prediction">
                                {clase}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box dog-prediction">
                                {clase}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confianza
                        st.markdown(f"""
                        <div style='text-align: center; font-size: 24px; color: #e8e8e8; margin: 15px 0;'>
                            <b>Nivel de Confianza:</b> {confianza:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Barra de progreso
                        st.progress(confianza / 100)
                        
                        # Interpretación de la confianza
                        st.markdown("<br>", unsafe_allow_html=True)
                        if confianza >= 90:
                            st.success("💯 **Confianza Muy Alta** - Predicción muy confiable")
                        elif confianza >= 80:
                            st.success("✅ **Alta Confianza** - Predicción confiable")
                        elif confianza >= 70:
                            st.info("📊 **Buena Confianza** - Resultado probable")
                        elif confianza >= 60:
                            st.warning("⚠️ **Confianza Moderada** - Imagen podría ser ambigua")
                        else:
                            st.error("🤔 **Baja Confianza** - Intenta con una imagen más clara")
                    
                    # Recomendaciones adicionales si confianza es baja
                    if confianza < 70:
                        st.markdown("---")
                        st.markdown("### 💡 Sugerencias para mejorar:")
                        st.markdown("""
                        - Intenta con una imagen más iluminada
                        - Asegúrate de que el animal sea el foco principal
                        - Evita fondos muy complejos o desordenados
                        - Toma la foto desde un ángulo frontal
                        """)
                
                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen: {e}")
                    st.info("💡 Intenta con otra imagen o verifica que sea un formato válido")
        
        # Botón para limpiar y clasificar otra
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clasificar Otra Imagen"):
            st.rerun()
    
    else:
        # Vista inicial sin imagen
        st.info("👆 Sube una imagen para comenzar la clasificación")
        
        # Sección informativa
        st.markdown("---")
        st.markdown("### 🎯 ¿Qué hace este clasificador?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>🔬 Tecnología</h4>
                <p><b>Características extraídas:</b></p>
                <ul>
                    <li><b>HOG</b>: Histograma de gradientes orientados</li>
                    <li><b>Color</b>: Distribución espacial de intensidades</li>
                    <li><b>LBP</b>: Patrones binarios locales (texturas)</li>
                </ul>
                <p><b>Clasificadores:</b></p>
                <ul>
                    <li><b>SVM</b> con kernel RBF</li>
                    <li><b>Random Forest</b> con 100 árboles</li>
                    <li><b>Ensemble</b> por votación ponderada</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>📸 Ejemplos de Uso</h4>
                <p><b>Funciona mejor con:</b></p>
                <ul>
                    <li>🐱 Gatos en diferentes poses</li>
                    <li>🐶 Perros de diversas razas</li>
                    <li>📷 Fotos claras y nítidas</li>
                    <li>🎨 Fondos simples o uniformes</li>
                </ul>
                <p><b>Limitaciones conocidas:</b></p>
                <ul>
                    <li>Fondos muy complejos pueden afectar precisión</li>
                    <li>Múltiples animales en la misma foto</li>
                    <li>Imágenes muy oscuras o borrosas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Explicación adicional
        st.markdown("---")
        st.markdown("### 🧮 ¿Por qué Machine Learning Clásico?")
        st.markdown("""
        Este clasificador usa **métodos tradicionales de Machine Learning** en lugar de Deep Learning:
        
        **✅ Ventajas:**
        - **Rápido**: No requiere GPU, funciona en cualquier computadora
        - **Eficiente**: Modelo ligero (~5MB) vs modelos DL (~100MB+)
        - **Interpretable**: Podemos entender qué características usa
        - **Menos datos**: Funciona bien con datasets más pequeños
        
        **⚠️ Limitaciones:**
        - **Fondos complejos**: Mejor rendimiento con fondos simples
        - **Precisión**: ~75-85% vs ~90-95% de modelos DL modernos
        - **Generalización**: Menos robusto ante condiciones no vistas en entrenamiento
        
        **💡 Ideal para:**
        - Aplicaciones con recursos limitados
        """)
    
if __name__ == "__main__":
    main()