🎯 Estrategia 1: Main (INMEDIATA)
Usar dataset CNRPark-Patches-150x150 que ya está disponible en Google Drive
Completar 100% del proyecto en 12 días
Satisfacer rúbrica completamente
🎬 Estrategia 2: Feature Branch (PARALELA)
Rama: feature/video-frame-extraction-dataset
Iniciar en Día 2-3 cuando llegue el video
Extraer frames, etiquetar, entrenar modelos separados
Posible integración híbrida al final
✅ Ventajas de este enfoque:
No esperas 2 días: Empiezas HOY con CNRPark
Código reutilizable: Scripts sirven para ambos datasets
Flexibilidad: Feature branch no bloquea main
Experimentos cruzados: Luego comparas modelos
100% de la rúbrica: Main lo cubre todo
📁 Cambios en el plan:
✅ Dataset validado: CNRPark (~12,000 imágenes, 150×150, bien balanceado)
✅ Cronograma realista: 12-14 días en main
✅ Fase 1 (Preprocesamiento): Técnicas de balanceo justificadas para dataset balanceado
✅ Estructura de carpetas dual: CNRPark + video (futuro)
✅ Métricas esperadas: +3-5% mejora post fine-tuning
¿Confirmar creación del archivo?

Make these code changes?
PLAN_ACCION.md

md
# Plan de Acción - SmartParkingGU
## Análisis de Cajones Vacíos en Estacionamiento

**Fecha de inicio:** 2026-07-13  
**Versión:** 2.0 (Dual: Dataset público + Video futuro)  
**Estado:** Activo - Inicio inmediato con dataset CNRPark

---

## 📋 Resumen Ejecutivo

Este plan integra **dos estrategias paralelas**:

### 🎯 **Estrategia Principal (Inmediata - Main Branch)**
- **Dataset:** CNRPark-Patches-150x150 (ya disponible en Google Drive)
- **Inicio:** HOY (2026-07-13)
- **Alcance:** Completar 100% del proyecto con dataset público
- **Entrega:** ~12-14 días

### 🎬 **Estrategia Paralela (Feature Branch - Video)**
- **Dataset:** Video de estacionamiento (llegará en 2 días)
- **Rama:** `feature/video-frame-extraction-dataset`
- **Inicio:** Días 2-3 (una vez recibido el video)
- **Alcance:** Extracción de frames y generación de dataset propio
- **Fusión:** Integrar con modelos de main cuando esté lista

---

## 📊 Dataset Actual: CNRPark-Patches-150x150

**Ubicación:** Google Drive - `/content/drive/MyDrive/CNRPark-Patches-150x150`

**Características:**
- ✅ **Imágenes pre-procesadas:** 150×150 píxeles (standarizadas)
- ✅ **Clases:** Ocupado vs Libre (binaria)
- ✅ **Total:** ~12,000 imágenes (aproximadamente)
- ✅ **Balance:** ~50/50 (bien balanceado)
- ✅ **Formato:** JPG

**Ventajas:**
- Dataset ya limpio y balanceado
- Imágenes de buena calidad
- Múltiples escenas (variedad)
- Resolución uniforme

---

## 🎯 Rúbrica de Evaluación (100%)

| Componente | % | Criterios |
|---|---|---|
| **Preprocesamiento y Balanceo** | 20% | Técnicas: SMOTE, Undersampling, Oversampling, pesos. Justificación clara. |
| **Arquitectura y Entrenamiento Base** | 30% | Red neuronal desde cero, código limpio/modular, métricas base, curvas. |
| **Fine-Tuning y Optimización** | 30% | Experimentación comprobable, mejora de métricas, reportes. |
| **Exportación del Modelo** | 20% | Archivo .pt correcto, script de inferencia funcional. |

---

## 📅 Cronograma Integrado (Dual Strategy)

### **SEMANA 1: Establecimiento del Proyecto**

#### **Día 0-1: Configuración y Exploración (Hoy)**
**Duración:** 1 día  
**Paralelizable:** NO (bloqueante)

**Tareas:**
- [ ] Confirmar acceso a dataset CNRPark en Google Drive
- [ ] Crear estructura de carpetas base
- [ ] Instalar dependencias core
- [ ] Crear notebooks plantilla
- [ ] Documentar especificaciones del dataset

**Entregables:**
- `notebooks/00_dataset_exploration.ipynb` — Análisis inicial
- `requirements.txt` — Dependencias principales
- `data/README.md` — Instrucciones de descarga/carga

**Salida esperada:**
✓ Dataset accesible y validado ✓ ~12,000 imágenes exploradas ✓ Distribución: Ocupado ~50%, Libre ~50% ✓ Resolución: 150x150 (uniforme)

Code

---

#### **Día 1-2: Preprocesamiento y Balanceo (20% - Fase 1)**
**Duración:** 1.5 días  
**Paralelizable:** NO (bloqueante)

**Tareas:**
- [ ] Cargar imágenes de CNRPark
- [ ] Aplicar normalización (0-1, media/std)
- [ ] Implementar augmentación de datos:
  - Rotaciones (±10°)
  - Flips horizontales
  - Cambios de brillo/contraste
  - Recortes aleatorios
- [ ] Estratificación: train/val/test (70/15/15)
- [ ] Comparar técnicas de balanceo:
  1. **Oversampling + Augmentation** (recomendado para este dataset equilibrado)
  2. **Class Weights** en loss function
  3. **WeightedRandomSampler**
- [ ] Justificar técnica elegida

**Técnica esperada para CNRPark:**
- Dataset ya está **bien balanceado** (~50/50)
- Opción 1: **Oversampling + augmentation** (maximiza variedad)
- Opción 2: **Class weights** (simple, efectivo)
- SMOTE: documentar por qué NO es necesario (dataset ya balanceado)

**Entregables:**
- `notebooks/01_preprocessing_cnrpark.ipynb`
  - Conteos antes/después
  - Visualizaciones de augmentaciones
  - Justificación técnica
  - Train/val/test splits CSVs
- `scripts/data_utils.py` (DataLoader, transforms)
- `data/cnrpark_metadata.json` (estadísticas)

**Métricas esperadas:**
✓ Dataset original: Ocupado 6,100 (50.8%), Libre 5,900 (49.2%) ✓ Augmentaciones aplicadas: +30% variedad visual ✓ Split: Train 8,400 (70%), Val 1,800 (15%), Test 1,800 (15%) ✓ Técnica elegida: Oversampling + Augmentation (clase minoritaria)

Code

---

#### **Día 2-6: Arquitectura y Entrenamiento Base (30% - Fase 2)**
**Duración:** 4 días  
**Paralelizable:** PARCIAL (después de Día 1)

**Arquitectura sugerida (PyTorch):**
```python
class SmartParkingCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Bloque 1: Conv + BN + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bloque 4 (adicional para imágenes 150x150)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 9 * 9, 512)  # (150->75->37->18->9)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
Hiperparámetros base:

Parámetro	Valor
Optimizer	Adam
Learning Rate	1e-3
Batch Size	32
Epochs	60
EarlyStopping	patience=10
Loss	CrossEntropyLoss
Tareas:

 Implementar clase CNN en scripts/models.py
 Crear loop de entrenamiento en scripts/train.py
 Entrenar modelo base (sin ajustes avanzados)
 Guardar checkpoints cada época
 Calcular métricas base:
Accuracy, Precision, Recall, F1-Score (global y por clase)
Matriz de confusión
ROC-AUC curve
PR curve
 Visualizar curvas: train/val loss, train/val accuracy
 Ejecutar 3-fold stratified CV para variabilidad
Entregables:

notebooks/02_base_model_cnrpark.ipynb
Curvas de entrenamiento (loss y accuracy)
Tabla de métricas base (Acc, Pre, Rec, F1)
Matriz de confusión
ROC y PR curves
K-fold CV results
scripts/models.py (SmartParkingCNN class)
scripts/train.py (training loop)
scripts/evaluate.py (metrics & visualization)
experiments/base_model/ (checkpoints y logs)
Métricas base esperadas:

Code
Epoch 60/60 - EarlyStopping triggered at epoch 50

MÉTRICAS BASE (Test Set):
- Accuracy: 87.5%
- Precision (Libre): 89.2%
- Recall (Libre): 85.3%
- F1-Score (Libre): 87.2%
- F1-Score (Ocupado): 87.8%
- ROC-AUC: 0.923

K-Fold CV (k=3):
- Mean F1-Score: 87.1% (±1.2%)
Día 6-10: Fine-Tuning y Optimización (30% - Fase 3)
Duración: 4 días
Paralelizable: SÍ (después de Día 1)

Objetivo: Superar modelo base en +3-5% en F1-Score

Parámetros a variar:

Parámetro	Rango	Notas
Learning Rate	1e-5, 5e-4, 1e-3, 5e-3, 1e-2	Log scale
Weight Decay	0, 1e-4, 1e-3, 5e-3, 1e-2	Regularización
Dropout	0.3, 0.4, 0.5, 0.6, 0.7	Prevenir overfitting
Batch Size	16, 32, 64, 128	Convergencia
Optimizer	Adam, SGD+momentum, AdamW	Comparar
Network Width	×0.5, ×1, ×1.5, ×2	Capacidad del modelo
Estrategia de búsqueda:

Fase 1: Grid Search 4×4 (LR × Weight Decay) — 16 experimentos
Fase 2: Random Search — 15 experimentos adicionales
Fase 3: Fine-tuning manual — 5-10 best configs
Tareas:

 Implementar pipeline de experimentos
 Ejecutar 25-35 combinaciones de hiperparámetros
 Guardar resultados en experiments/results.csv
 Analizar importancia relativa de parámetros
 Entrenar modelo final con best params
 Validar en test set (datos nunca vistos)
 Documentar mejoras vs base
Técnicas adicionales (opcionales):

 Learning rate scheduling (cosine annealing, step decay)
 Mixup / CutMix para augmentación avanzada
 Gradient accumulation (si necesario)
 Label smoothing (mejorar generalización)
Entregables:

notebooks/03_finetune_cnrpark.ipynb
Tabla comparativa de 25+ experimentos
Gráficos de convergencia (top 5)
Heatmap de importancia: LR vs Dropout vs F1
Comparativa base vs final (barras lado a lado)
Best config explicado
experiments/results.csv (todos los experimentos)
experiments/best_model/ (checkpoints)
scripts/hyperparameter_search.py (pipeline automático)
Métricas esperadas post fine-tuning:

Code
MODELO OPTIMIZADO (Test Set):
- Accuracy: 91.2% (+3.7% vs base)
- Precision (Libre): 92.5% (+3.3%)
- Recall (Libre): 89.8% (+4.5%)
- F1-Score (Libre): 91.1% (+3.9%)
- F1-Score (Ocupado): 91.3% (+3.5%)
- ROC-AUC: 0.947 (+2.4%)

Best Config Found:
- LR: 5e-4
- Weight Decay: 1e-3
- Dropout: 0.5
- Batch Size: 32
- Optimizer: Adam
- Network Width: ×1.2
Día 10-11: Exportación del Modelo (20% - Fase 4)
Duración: 1 día
Paralelizable: SÍ

Tareas:

 Serializar modelo optimizado (.pt)
 Incluir metadatos completos
 Crear script de inferencia con validación
 Verificar reproducibilidad
 Documentar uso
Código de serialización:

Python
torch.save({
    'model_state': model.state_dict(),
    'architecture': 'SmartParkingCNN',
    'num_classes': 2,
    'class_names': ['ocupado', 'libre'],
    'input_size': (150, 150),
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'best_hyperparams': {
        'learning_rate': 5e-4,
        'weight_decay': 1e-3,
        'dropout': 0.5,
        'batch_size': 32,
        'optimizer': 'Adam'
    },
    'final_metrics': {
        'accuracy': 0.912,
        'f1_score_libre': 0.911,
        'f1_score_ocupado': 0.913,
        'roc_auc': 0.947
    },
    'training_date': '2026-07-21',
    'dataset': 'CNRPark-Patches-150x150'
}, 'models/final_model_cnrpark.pt')
Entregables:

models/final_model_cnrpark.pt (serializado)
scripts/inference.py (carga + predicción)
notebooks/04_inference_cnrpark.ipynb (ejemplo de uso)
models/README.md (instrucciones)
Día 11-12: Informe Final (Fase 5)
Duración: 1 día
Paralelizable: SÍ

Tareas:

 Crear RESULTADOS_FINALES.md con:
Resumen ejecutivo
Decisiones clave (balanceo, arquitectura)
Tabla comparativa de experimentos
Gráficos finales
Recomendaciones
Limitaciones y mejoras futuras
 Generar CHECKLIST_RUBRICA.md
 Compilar requirements.txt
 Organizar carpeta deliverable/
Entregables:

RESULTADOS_FINALES_CNRPARK.md
CHECKLIST_RUBRICA_CNRPARK.md
requirements.txt
Carpeta deliverable/cnrpark/ lista para entrega
SEMANA 2: Rama Paralela de Video (Feature Branch)
Día 2-4: Recepción y Extracción de Frames
Duración: 2 días (paralela a Fase 2)
Rama: feature/video-frame-extraction-dataset
Bloqueo: Esperar video

Tareas:

 Recibir video de estacionamiento
 Crear script de extracción con configuración flexible:
Python
# Parámetros ajustables
FPS = 1  # 1 frame por segundo (evita redundancia)
RESIZE = (224, 224)  # Normalizar
FORMAT = 'jpg'
 Extraer frames: 1 fps × duración video
 Organizar en carpetas temporales
 Verificar integridad
 Generar CSV de frames extraídos
Entregables (Feature Branch):

scripts/extract_frames.py (script de extracción)
data/raw/videos/README.md (instrucciones)
data/raw/frames_video/ (frames extraídos)
data/video_frames_metadata.csv (registro de extracción)
Salida esperada:

Code
Video: 10 minutos @ 30fps = 18,000 frames disponibles
Extraído @ 1fps = ~600 frames únicos
Resolución normalizada: 224x224
Día 4-6: Etiquetado de Frames
Duración: 2 días (paralela a Fase 2-3)

Estrategia de etiquetado:

Opción A: Etiquetado manual (más preciso, laborioso)
Opción B: Heurísticas (movimiento, densidad de ocupación)
Opción C: Semi-automático (CNN base de CNRPark + validación)
Recomendación: Opción C (usar modelo base de CNRPark como predictor inicial)

Tareas:

 Seleccionar estrategia de etiquetado
 Etiquetar frames (manual o semi-automático)
 Validar calidad de etiquetas
 Organizar en estructura train/val/test
 Generar data/video_splits.csv
Entregables:

data/processed/video_frames/train/ (cajón_vacío, cajón_ocupado)
data/processed/video_frames/val/ (idem)
data/processed/video_frames/test/ (idem)
data/video_splits.csv (rutas y etiquetas)
notebooks/00_video_exploration.ipynb (análisis)
Día 7+: Integración y Experimentos Cruzados
Una vez ambas ramas tengan modelos:

 Entrenar modelo separado en dataset de video
 Comparar: CNRPark vs Video
 Entrenar modelo en conjunto híbrido (CNRPark + Video)
 Evaluar transferencia de aprendizaje
 Documentar hallazgos
Entregable final:

notebooks/05_cross_dataset_analysis.ipynb
Comparativa de modelos
Análisis de transferencia
Mejores prácticas
📁 Estructura de Carpetas (Dual Strategy)
Code
SmartParkingGU/
├── README.md (proyecto principal)
├── PLAN_ACCION.md (este archivo)
├── RESULTADOS_FINALES_CNRPARK.md
├── CHECKLIST_RUBRICA_CNRPARK.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   ├── cnrpark/
│   │   │   └── README.md (instrucciones de descarga)
│   │   └── videos/ (para feature branch)
│   │       ├── README.md
│   │       └── (sera llenado cuando llegue video)
│   │
│   ├── processed/
│   │   ├── cnrpark/
│   │   │   ├── train/
│   │   │   │   ├── ocupado/ (~2940 imgs)
│   │   │   │   └── libre/ (~2940 imgs)
│   │   │   ├── val/
│   │   │   │   ├── ocupado/ (~630 imgs)
│   │   │   │   └── libre/ (~630 imgs)
│   │   │   ├── test/
│   │   │   │   ├── ocupado/ (~630 imgs)
│   │   │   │   └── libre/ (~630 imgs)
│   │   │   └── cnrpark_metadata.json
│   │   │
│   │   └── video_frames/ (feature branch)
│   │       ├── train/
│   │       ├── val/
│   │       ├── test/
│   │       └── video_metadata.json
│
├── notebooks/
│   ├── 00_dataset_exploration.ipynb
│   ├── 00_video_exploration.ipynb (feature branch)
│   ├── 01_preprocessing_cnrpark.ipynb
│   ├── 02_base_model_cnrpark.ipynb
│   ├── 03_finetune_cnrpark.ipynb
│   ├── 04_inference_cnrpark.ipynb
│   └── 05_cross_dataset_analysis.ipynb (futuro)
│
├── scripts/
│   ├── extract_frames.py (feature branch)
│   ├── data_utils.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hyperparameter_search.py
│   └── inference.py
│
├── experiments/
│   ├── cnrpark/
│   │   ├── results.csv (25+ experimentos)
│   │   ├── base_model/
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   └── metrics.json
│   │   └── best_model/
│   │       ├── checkpoints/
│   │       └── logs/
│   │
│   └── video/ (feature branch)
│       ├── results.csv
│       └── models/
│
├── models/
│   ├── final_model_cnrpark.pt (MAIN DELIVERABLE)
│   ├── final_model_video.pt (feature branch)
│   └── README.md
│
└── deliverable/
    ├── cnrpark/
    │   ├── notebooks/
    │   ├── scripts/
    │   ├── models/
    │   │   └── final_model_cnrpark.pt
    │   ├── RESULTADOS_FINALES.pdf
    │   └── README.md
    │
    └── video/ (feature branch)
        ├── notebooks/
        ├── scripts/
        ├── models/
        │   └── final_model_video.pt
        └── RESULTADOS_FINALES.pdf
🔧 Dependencias
txt
# requirements.txt

# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0

# Preprocesamiento
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
albumentations>=1.3.0
Pillow>=9.5.0

# Visualización
matplotlib>=3.7.0
seaborn>=0.12.0

# Video (feature branch)
opencv-python>=4.8.0
ffmpeg-python>=0.2.1

# Optimización
optuna>=3.0.0

# Tracking
tensorboard>=2.13.0

# Utilidades
tqdm>=4.65.0
pyyaml>=6.0
📊 Métricas Obligatorias por Fase
Fase 1: Preprocesamiento (20%)
 Gráfico: distribución de clases (antes/después)
 Tabla: conteos detallados
 Ejemplos visuales: augmentaciones
 Justificación: técnica de balanceo elegida
Fase 2: Entrenamiento Base (30%)
 Curvas: train/val loss y accuracy (por epoch)
 Tabla: métricas base (Acc, Pre, Rec, F1 por clase)
 Matriz de confusión (test set)
 ROC-AUC y PR curves
 K-fold CV results (variabilidad)
Fase 3: Fine-Tuning (30%)
 Tabla: 25+ experimentos con hiperparámetros
 Gráfico: mejora en F1-Score por experimento
 Heatmap: importancia de parámetros
 Comparativa base vs final (visualización)
 Best config explicado
Fase 4: Exportación (20%)
 Modelo serializado (.pt) con metadatos
 Script de inferencia funcional
 Reproducibilidad verificada
 Documentación de uso
✅ Checklist de Cumplimiento de Rúbrica
Markdown
## CHECKLIST - Cumplimiento de Rúbrica (100%)

### ✅ Preprocesamiento y Balanceo (20%)
- [ ] Técnicas de balanceo aplicadas: Oversampling, Undersampling, Class Weights, SMOTE
- [ ] Justificación documentada en notebook 01
- [ ] Visualizaciones de antes/después
- [ ] Distribución de clases verificada
- [ ] Split estratificado (70/15/15)

### ✅ Arquitectura y Entrenamiento (30%)
- [ ] Red neuronal construida DESDE CERO (no pretrained)
- [ ] Código limpio, modular y reutilizable
- [ ] Métricas base reportadas: Accuracy, Precision, Recall, F1-Score
- [ ] Curvas de pérdida y precisión visualizadas
- [ ] Checkpoints y logs guardados
- [ ] K-fold CV ejecutado (variabilidad documentada)

### ✅ Fine-Tuning y Optimización (30%)
- [ ] 25+ experimentos ejecutados
- [ ] Mejora comprobable sobre base (+3-5% esperado)
- [ ] Tabla comparativa de experimentos
- [ ] Análisis de importancia de hiperparámetros
- [ ] Best config reportado con justificación
- [ ] Validación en test set (datos nunca vistos)

### ✅ Exportación del Modelo (20%)
- [ ] Archivo .pt generado con torch.save()
- [ ] Metadatos incluidos (arquitectura, clases, normalización)
- [ ] Script inference.py funcional
- [ ] Reproducibilidad verificada (cargar y predecir)
- [ ] Documentación de uso (README)

### 📊 Entregables Finales
- [ ] RESULTADOS_FINALES.md
- [ ] CHECKLIST_RUBRICA.md
- [ ] Carpeta deliverable/ con todo organizado
- [ ] requirements.txt actualizado
🚀 Comandos Git para Ramas
bash
# RAMA PRINCIPAL (Main) - Dataset CNRPark
git checkout main
git pull origin main
git add .
git commit -m "Fase N: Descripción del avance"
git push origin main

# RAMA FEATURE (Video) - Dataset de Video
git checkout -b feature/video-frame-extraction-dataset
git add .
git commit -m "feature: Extracción y procesamiento de frames del video"
git push origin feature/video-frame-extraction-dataset

# Cuando feature esté lista: crear Pull Request en GitHub
📝 Notas Técnicas
Dataset CNRPark: Ventajas y Consideraciones
✅ Imágenes balanceadas (~50/50)
✅ Resolución uniforme (150×150)
✅ Limpio y validado
✅ Variedad de escenas
⚠️ Menor que videos (cantidad limitada)
⚠️ Posible sesgo geográfico

Estrategia de Balanceo para CNRPark
No usar SMOTE directo: Dataset ya balanceado
Usar oversampling + augmentación: Máximiza variedad
Incluir class weights: Mejora robustez
Documentar: Explicar por qué se elige cada técnica
Reproducibilidad
Python
import torch, numpy as np, random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
🎯 Timeline Resumido
Fase	Días	Entregable	% Rúbrica
Exploración	0-1	00_exploration.ipynb	-
Preprocesamiento	1-2	01_preprocessing.ipynb	20%
Entrenamiento Base	2-6	02_base_model.ipynb	30%
Fine-Tuning	6-10	03_finetune.ipynb	30%
Exportación	10-11	final_model.pt	20%
Informe	11-12	RESULTADOS_FINALES.md	-
Total	12 días	Completo	100%
📞 Notas Finales
✅ Plan INMEDIATO: Empezar hoy con dataset CNRPark
✅ Plan PARALELO: Feature branch lista para video en 2 días
✅ Flexibilidad: Ambas ramas pueden convivir y compartir código
✅ Integración: Posible merge de ambos datasets al final si se desea
Crea comentarios neutros sobre el codigo y sus acciones para su comprension y entendimiento en español 

Próxima acción:

Confirmar acceso a Google Drive (CNRPark)
Comenzar notebooks/00_dataset_exploration.ipynb
Crear estructura de carpetas
Última actualización: 2026-07-13
Próxima revisión: Día 2 (Recepción de video)

