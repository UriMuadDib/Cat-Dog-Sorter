import os
import time
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# CONFIGURACIÓN MEJORADA
# -------------------------------------------------
RUTA_TRAIN = 'Data/training_set'
RUTA_TEST = 'Data/test_set'

# Aumentar resolución para capturar más detalles
IMG_HEIGHT = 128
IMG_WIDTH = 128


# -------------------------------------------------
# FUNCIÓN MEJORADA: CARGAR Y PROCESAR IMÁGENES
# -------------------------------------------------
def cargar_y_procesar(ruta_base, max_images_per_class=None):
    features = []
    labels = []
    clases = ['cats', 'dogs']

    print(f"Iniciando carga desde: {ruta_base}")

    for categoria in clases:
        path_categoria = os.path.join(ruta_base, categoria)
        etiqueta = 0 if categoria == 'cats' else 1

        archivos = [f for f in os.listdir(path_categoria) if f.endswith('.jpg')]
        
        # Limitar cantidad si se especifica (útil para pruebas rápidas)
        if max_images_per_class:
            archivos = archivos[:max_images_per_class]
        
        print(f"  Procesando {len(archivos)} imágenes de {categoria}...")

        for archivo in archivos:
            img_path = os.path.join(path_categoria, archivo)
            try:
                # 1. Cargar imagen
                img = imread(img_path)
                
                # 2. Convertir a escala de grises si es necesario
                if len(img.shape) == 3:
                    img_gray = rgb2gray(img)
                else:
                    img_gray = img

                # 3. Redimensionar con mejor calidad
                img_resized = resize(img_gray, (IMG_HEIGHT, IMG_WIDTH), 
                                    anti_aliasing=True, preserve_range=True)

                # 4. Extraer características HOG mejoradas
                fd = hog(
                    img_resized,
                    orientations=12,  # Más orientaciones para capturar más detalles
                    pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3),  # Bloques más grandes para mejor normalización
                    block_norm='L2-Hys',  # Mejor método de normalización
                    feature_vector=True
                )

                features.append(fd)
                labels.append(etiqueta)

            except Exception as e:
                print(f"Error en {archivo}: {e}")

    return np.array(features), np.array(labels)


# -------------------------------------------------
# FUNCIÓN: AGREGAR DATA AUGMENTATION (OPCIONAL)
# -------------------------------------------------
def augmentar_datos(X, y):
    """
    Duplica el dataset con imágenes espejadas (horizontal flip simulado en features)
    Esto ayuda a mejorar la generalización
    """
    print("Aplicando data augmentation...")
    # Nota: Para HOG, el flip horizontal cambia la distribución de gradientes
    # Aquí simplemente duplicamos los datos originales
    # En un enfoque más avanzado, procesarías las imágenes volteadas
    X_aug = np.vstack([X, X])
    y_aug = np.hstack([y, y])
    return X_aug, y_aug


start_time = time.time()

# -------- FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS (TRAIN) --------
print("\n" + "="*60)
print("FASE 1: Extracción de Características (Training)")
print("="*60)
X_train, y_train = cargar_y_procesar(RUTA_TRAIN)
print(f"\n✓ Vector de características (train): {X_train.shape}")
print(f"  - Muestras de gatos: {np.sum(y_train == 0)}")
print(f"  - Muestras de perros: {np.sum(y_train == 1)}")

# -------- FASE 2: NORMALIZACIÓN --------
print("\n" + "="*60)
print("FASE 2: Normalización de características")
print("="*60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("✓ Datos normalizados (media=0, std=1)")

# -------- FASE 3: ENTRENAMIENTO SVM CON MEJORES HIPERPARÁMETROS --------
print("\n" + "="*60)
print("FASE 3: Entrenando SVM con kernel RBF")
print("="*60)

# Kernel RBF generalmente funciona mejor que linear para imágenes
# C controla el trade-off entre margen y clasificación correcta
# gamma controla la influencia de cada ejemplo de entrenamiento
svm_model = SVC(
    kernel='rbf',  # Cambio clave: RBF captura mejor patrones no lineales
    C=10,  # Regularización
    gamma='scale',  # Ajuste automático basado en características
    random_state=42,
    cache_size=500  # Más cache para entrenamiento más rápido
)

print("Entrenando modelo (esto puede tomar un momento)...")
svm_model.fit(X_train_scaled, y_train)
print("✓ Modelo entrenado!")

# Evaluar en entrenamiento para detectar overfitting
y_train_pred = svm_model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"  Precisión en entrenamiento: {train_acc * 100:.2f}%")

# -------- FASE 4: EVALUACIÓN (TEST) --------
print("\n" + "="*60)
print("FASE 4: Evaluación en conjunto de prueba")
print("="*60)
X_test, y_test = cargar_y_procesar(RUTA_TEST)
X_test_scaled = scaler.transform(X_test)

y_pred = svm_model.predict(X_test_scaled)

# -------- RESULTADOS DETALLADOS --------
print("\n" + "="*60)
print("RESULTADOS FINALES")
print("="*60)

test_acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Precisión en TEST: {test_acc * 100:.2f}%")
print(f"📊 Precisión en TRAIN: {train_acc * 100:.2f}%")

if train_acc - test_acc > 0.15:
    print("\n⚠️  Advertencia: Posible overfitting detectado")
    print("   (Gran diferencia entre train y test)")

print("\n" + "-"*60)
print("Reporte de Clasificación Detallado:")
print("-"*60)
print(classification_report(y_test, y_pred, target_names=['Gato', 'Perro']))

# Matriz de confusión
print("\n" + "-"*60)
print("Matriz de Confusión:")
print("-"*60)
cm = confusion_matrix(y_test, y_pred)
print(f"                  Predicho")
print(f"                  Gato  Perro")
print(f"Real    Gato      {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"        Perro     {cm[1,0]:4d}  {cm[1,1]:4d}")

# -------- FASE 5: GUARDAR MODELO --------
print("\n" + "="*60)
print("FASE 5: Guardando modelo")
print("="*60)
with open("modelo_svm_mascotas_mejorado.pkl", "wb") as f:
    pickle.dump({
        "model": svm_model,
        "scaler": scaler,
        "config": {
            "img_height": IMG_HEIGHT,
            "img_width": IMG_WIDTH,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        }
    }, f)

print("✓ Modelo guardado como: modelo_svm_mascotas_mejorado.pkl")

tiempo_total = (time.time() - start_time) / 60
print(f"\n⏱️  Tiempo total de ejecución: {tiempo_total:.2f} minutos")
print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)