import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time

# --- CONFIGURACIÓN ---
# Rutas a tus carpetas
RUTA_TRAIN = 'Data/training_set'
RUTA_TEST = 'Data/test_set'

# Tamaño reducido para que HOG no tarde horas (64x64 es estándar para pruebas rápidas)
IMG_HEIGHT = 64
IMG_WIDTH = 64

def cargar_y_procesar(ruta_base):
    features = []
    labels = []
    clases = ['cats', 'dogs'] # 0: Gato, 1: Perro
    
    print(f"Iniciando carga desde: {ruta_base}")
    
    for categoria in clases:
        path_categoria = os.path.join(ruta_base, categoria)
        etiqueta = 0 if categoria == 'cats' else 1
        
        # Leemos los archivos
        archivos = [f for f in os.listdir(path_categoria) if f.endswith('.jpg')]
        print(f"  Procesando {len(archivos)} imágenes de {categoria}...")
        
        for archivo in archivos:
            img_path = os.path.join(path_categoria, archivo)
            try:
                # 1. Cargar imagen
                img = imread(img_path)
                
                # 2. Redimensionar (crucial para que todos los vectores sean iguales)
                img_resized = resize(img, (IMG_HEIGHT, IMG_WIDTH))
                
                # 3. Extraer características HOG
                # pixels_per_cell=(8,8): Agrupa píxeles en bloques de 8x8
                # orientations=9: Busca bordes en 9 direcciones
                fd = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), channel_axis=-1)
                
                features.append(fd)
                labels.append(etiqueta)
            except Exception as e:
                print(f"Error en {archivo}: {e}")

    return np.array(features), np.array(labels)

# --- FLUJO PRINCIPAL ---
start_time = time.time()

# 1. Cargar y extraer características (Training)
print("--- FASE 1: Extracción de Características (Training) ---")
X_train, y_train = cargar_y_procesar(RUTA_TRAIN)
print(f"Vector de características: {X_train.shape}")

# 2. Entrenar el SVM
print("\n--- FASE 2: Entrenando SVM (Esto puede tardar unos minutos) ---")
# Kernel 'linear' suele funcionar mejor para vectores HOG grandes
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
print("¡Modelo entrenado!")

# 3. Evaluar (Test)
print("\n--- FASE 3: Evaluación (Test) ---")
X_test, y_test = cargar_y_procesar(RUTA_TEST)

# Predicción
y_pred = svm_model.predict(X_test)

# Reporte
print("\nRESULTADOS FINALES:")
print(f"Precisión (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nReporte Detallado:")
print(classification_report(y_test, y_pred, target_names=['Gato', 'Perro']))

print(f"Tiempo total: {(time.time() - start_time)/60:.1f} minutos")
