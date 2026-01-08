import os
import time
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RUTA_TRAIN = 'Data/training_set'
RUTA_TEST = 'Data/test_set'

IMG_HEIGHT = 128
IMG_WIDTH = 128

# EXTRACCIÓN DE CARACTERÍSTICAS MÚLTIPLES
def extraer_caracteristicas_avanzadas(img):
    """
    Combina múltiples descriptores para capturar diferentes aspectos de la imagen:
    1. HOG: Captura formas y bordes
    2. Color Histogram: Captura distribución de colores
    3. LBP: Captura texturas locales
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
    
    # 2. Color Histogram (distribución de intensidades)
    # Dividir imagen en 4 regiones para capturar información espacial
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
        np.array(histograms) / np.sum(histograms),  # Normalizar histograma
        lbp_hist / np.sum(lbp_hist)  # Normalizar LBP
    ])
    
    return features_combined


def cargar_y_procesar_mejorado(ruta_base, max_images_per_class=None):
    """Carga imágenes y extrae características avanzadas"""
    features = []
    labels = []
    clases = ['cats', 'dogs']

    print(f"📁 Cargando desde: {ruta_base}")

    for categoria in clases:
        path_categoria = os.path.join(ruta_base, categoria)
        etiqueta = 0 if categoria == 'cats' else 1

        archivos = [f for f in os.listdir(path_categoria) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_images_per_class:
            archivos = archivos[:max_images_per_class]
        
        print(f"  🔄 Procesando {len(archivos)} imágenes de {categoria}...")

        for idx, archivo in enumerate(archivos):
            if idx % 500 == 0 and idx > 0:
                print(f"     Progreso: {idx}/{len(archivos)}")
            
            img_path = os.path.join(path_categoria, archivo)
            try:
                img = imread(img_path)
                
                # Convertir a escala de grises
                if len(img.shape) == 3:
                    img_gray = rgb2gray(img)
                else:
                    img_gray = img

                # Redimensionar con anti-aliasing
                img_resized = resize(img_gray, (IMG_HEIGHT, IMG_WIDTH), 
                                   anti_aliasing=True, preserve_range=True)

                # Normalizar a rango [0, 1]
                img_resized = img_resized / 255.0

                # Extraer características avanzadas
                caracteristicas = extraer_caracteristicas_avanzadas(img_resized)
                
                features.append(caracteristicas)
                labels.append(etiqueta)

            except Exception as e:
                print(f"     ⚠️ Error en {archivo}: {e}")

    return np.array(features), np.array(labels)


# ENTRENAMIENTO CON ENSEMBLE (MEJORA LA ROBUSTEZ)
print("\n" + "="*70)
print("🚀 ENTRENAMIENTO MEJORADO - CLASIFICADOR CLÁSICO")
print("="*70)

start_time = time.time()

# -------- FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS --------
print("\n📊 FASE 1: Extracción de Características Avanzadas")
print("-"*70)
X_train, y_train = cargar_y_procesar_mejorado(RUTA_TRAIN)
print(f"\n✅ Características extraídas: {X_train.shape}")
print(f"   • Gatos: {np.sum(y_train == 0)} muestras")
print(f"   • Perros: {np.sum(y_train == 1)} muestras")

# -------- FASE 2: NORMALIZACIÓN --------
print("\n⚙️ FASE 2: Normalización")
print("-"*70)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("✅ Datos estandarizados (μ=0, σ=1)")

# -------- FASE 3: ENTRENAMIENTO ENSEMBLE --------
print("\n🤖 FASE 3: Entrenamiento de Ensemble")
print("-"*70)

# Modelo 1: SVM con kernel RBF (bueno para patrones no lineales)
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,  # Necesario para ensemble
    random_state=42,
    cache_size=1000
)

# Modelo 2: Random Forest (robusto a ruido y outliers)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Ensemble: Combina ambos modelos (voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', rf_model)
    ],
    voting='soft',  # Usa probabilidades
    weights=[1.2, 0.8]  # SVM tiene más peso
)

print("🔄 Entrenando ensemble (SVM + Random Forest)...")
ensemble_model.fit(X_train_scaled, y_train)
print("✅ Ensemble entrenado correctamente!")

# Evaluar en entrenamiento
y_train_pred = ensemble_model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"   📈 Precisión en entrenamiento: {train_acc * 100:.2f}%")

# -------- FASE 4: EVALUACIÓN EN TEST --------
print("\n🎯 FASE 4: Evaluación en Conjunto de Prueba")
print("-"*70)
X_test, y_test = cargar_y_procesar_mejorado(RUTA_TEST)
X_test_scaled = scaler.transform(X_test)

y_pred = ensemble_model.predict(X_test_scaled)

# -------- RESULTADOS FINALES --------
print("\n" + "="*70)
print("📊 RESULTADOS FINALES")
print("="*70)

test_acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Precisión en TEST:  {test_acc * 100:.2f}%")
print(f"📚 Precisión en TRAIN: {train_acc * 100:.2f}%")

diferencia = train_acc - test_acc
if diferencia > 0.15:
    print(f"\n⚠️  ADVERTENCIA: Overfitting detectado (diferencia: {diferencia*100:.1f}%)")
elif diferencia < 0.05:
    print(f"\n✅ Excelente generalización (diferencia: {diferencia*100:.1f}%)")
else:
    print(f"\n👍 Buena generalización (diferencia: {diferencia*100:.1f}%)")

print("\n" + "-"*70)
print("📋 Reporte Detallado por Clase:")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=['Gato 🐱', 'Perro 🐶']))

print("\n" + "-"*70)
print("🔢 Matriz de Confusión:")
print("-"*70)
cm = confusion_matrix(y_test, y_pred)
print(f"\n              Predicho")
print(f"              Gato   Perro")
print(f"Real  Gato    {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"      Perro   {cm[1,0]:4d}   {cm[1,1]:4d}")

# Calcular métricas adicionales
tp = cm[1,1]  # True Positives (Perros correctos)
tn = cm[0,0]  # True Negatives (Gatos correctos)
fp = cm[0,1]  # False Positives (Gatos predichos como perros)
fn = cm[1,0]  # False Negatives (Perros predichos como gatos)

print(f"\n📊 Métricas Adicionales:")
print(f"   • Sensibilidad (Recall Perros): {tp/(tp+fn)*100:.1f}%")
print(f"   • Especificidad (Recall Gatos):  {tn/(tn+fp)*100:.1f}%")

# -------- FASE 5: GUARDAR MODELO --------
print("\n💾 FASE 5: Guardando Modelo")
print("-"*70)

modelo_data = {
    "model": ensemble_model,
    "scaler": scaler,
    "config": {
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "version": "classic_ensemble",
        "feature_dims": X_train.shape[1]
    }
}

with open("modelo.pkl", "wb") as f:
    pickle.dump(modelo_data, f)

print("✅ Modelo guardado: modelo.pkl")
print(f"   Tamaño: {os.path.getsize('modelo.pkl') / (1024*1024):.1f} MB")

tiempo_total = (time.time() - start_time) / 60
print(f"\n⏱️  Tiempo total: {tiempo_total:.2f} minutos")
print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)