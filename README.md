# SmartParkingGU - Sistema de Deteccion de Ocupacion de Estacionamiento

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/KaraIbr/SmartParkingGU/actions/workflows/ci.yml/badge.svg)](https://github.com/KaraIbr/SmartParkingGU/actions)

Sistema basado en Vision Artificial y Aprendizaje Profundo para detectar automaticamente la ocupacion de espacios de estacionamiento utilizando imagenes de camaras de vigilancia.

## Diagramas

Ver [DIAGRAMAS.md](docs/DIAGRAMAS.md) para los diagramas de flujo del proyecto (arquitectura CNN, pipeline de datos, modulos, inferencia, entorno de ejecucion).

## Quick Start

```bash
# Clonar repositorio
git clone https://github.com/KaraIbr/SmartParkingGU.git
cd SmartParkingGU

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar tests
pytest tests/ -v
```

## Estructura del Proyecto

```
SmartParkingGU/
├── src/                        # Modulos del pipeline ML
│   ├── models.py              # Arquitectura SmartParkingCNN
│   ├── train.py               # Loop de entrenamiento + EarlyStopping
│   ├── evaluate.py            # Metricas y visualizaciones
│   ├── data.py                # Dataset personalizado
│   ├── inference.py           # Prediccion con modelo serializado
│   ├── hyperparameter_search.py
│   └── utils/                 # Utilidades
│       ├── extract_frames.py  # Extraccion de frames de video
│       ├── analyze_videos.py  # Analisis CV de frames
│       └── colab_connect.py   # Verificacion GPU/Colab
├── notebooks/                  # Notebooks de Colab
│   ├── 01_dataset_overview.ipynb
│   ├── 02_base_model_cnrpark.ipynb
│   └── 03_finetune_cnrpark.ipynb
├── tests/                      # Tests unitarios (pytest)
├── data/                       # Datos
│   └── raw/
│       ├── videos/            # Videos de estacionamiento (.mp4)
│       └── frames/            # Frames extraidos de videos
├── outputs/                    # Resultados generados
│   ├── models/                # Modelos entrenados (.pt)
│   ├── reports/               # Metricas y reportes
│   └── figures/               # Figuras y graficas
├── docs/                       # Documentacion adicional
│   ├── plan.md                # Plan de accion
│   └── DIAGRAMAS.md           # Diagramas de flujo
├── pyproject.toml              # Configuracion del proyecto
└── requirements.txt            # Dependencias
```

## Metodologia

1. **Exploracion de datos (EDA)** - Analisis del dataset CNRPark-Patches-150x150
2. **Preprocesamiento** - Normalizacion, division train/val/test (70/15/15)
3. **Entrenamiento base** - CNN desde cero en CNRPark
4. **Fine-tuning** - Optimizacion de hiperparametros
5. **Video** - Extraccion de frames y fine-tuning con datos propios
6. **Evaluacion** - Metricas completas + comparacion de modelos



# Integrantes

| Nombre    | Rol                                    |
| --------- | -------------------------------------- |
| Karina Ibarra  | Análisis Exploratorio de Datos (EDA)   |
| Diego Cabrera Ramírez   | Preparación y organización del dataset |
| Karina Ibarra y  Diego Cabrera Ramírez| Entrenamiento y evaluación del modelo  |

**Institución:** Global University

**Asignatura:** Machine Learning e Inteligencia Artificial

**Profesor:** Jorce 

**Periodo:** 2026


# Planteamiento del Problema

En estacionamientos cerrados de centros comerciales, universidades, hospitales y edificios corporativos, uno de los principales problemas operativos es la dificultad para localizar espacios disponibles.

Los conductores suelen recorrer múltiples pasillos antes de encontrar un cajón libre, generando:

* Tiempos prolongados de búsqueda.
* Congestión vehicular interna.
* Incremento en el consumo de combustible.
* Mayor emisión de contaminantes.
* Experiencias negativas para los usuarios.

La falta de información en tiempo real sobre la disponibilidad de espacios ocasiona además una utilización ineficiente de la infraestructura existente.

Actualmente, muchas soluciones comerciales utilizan sensores físicos instalados individualmente en cada cajón de estacionamiento. Aunque estos sistemas ofrecen buenos niveles de precisión, implican elevados costos de instalación, mantenimiento y expansión.

Por ello, surge la necesidad de desarrollar una solución más flexible, escalable y económicamente viable basada en visión artificial y aprendizaje profundo.


# Justificación Técnica y Económica

La propuesta utiliza cámaras de vigilancia y modelos de Deep Learning para identificar automáticamente si un espacio de estacionamiento se encuentra libre u ocupado.

## Reducción de Costos de Infraestructura

Los sistemas tradicionales requieren:

* Un sensor por cada espacio.
* Cableado o infraestructura inalámbrica.
* Módulos de comunicación.
* Fuentes de alimentación.
* Procesos de instalación y calibración.

Por ejemplo, para un estacionamiento de 51 espacios serían necesarios al menos 51 sensores individuales.

En contraste, una o pocas cámaras estratégicamente ubicadas pueden supervisar simultáneamente decenas de espacios mediante técnicas de visión artificial.

## Menor Mantenimiento Operativo

Los sensores físicos están expuestos a:

* Humedad.
* Polvo.
* Vibraciones.
* Golpes.
* Desgaste natural.

La sustitución de sensores defectuosos implica costos recurrentes de mantenimiento.

Por otro lado, una cámara correctamente instalada requiere menos intervenciones físicas y permite mejorar continuamente el sistema mediante actualizaciones de software.

## Mayor Escalabilidad

Cuando se amplía la capacidad de un estacionamiento, los sistemas tradicionales requieren la instalación de sensores adicionales para cada nuevo cajón.

Con visión artificial, la expansión puede lograrse mediante:

* Actualización del modelo.
* Incorporación de nuevas cámaras.
* Reentrenamiento del sistema.

Esto reduce significativamente los costos de crecimiento de la solución.



# Objetivos

## Objetivo General

Desarrollar un sistema inteligente capaz de detectar automáticamente la ocupación de espacios de estacionamiento mediante técnicas de visión artificial y aprendizaje profundo.

## Objetivos Específicos

* Analizar las características del conjunto de datos seleccionado.
* Realizar un análisis exploratorio de datos (EDA).
* Evaluar el balance entre clases.
* Preparar conjuntos de entrenamiento, validación y prueba.
* Entrenar un modelo de clasificación binaria.
* Evaluar el desempeño utilizando métricas estándar de Machine Learning.
* Analizar fortalezas, limitaciones y posibles mejoras futuras.



# Dataset

## Nombre del Dataset

CNRPark+EXT: A Dataset for Visual Occupancy Detection of Parking Lots

## Autores

* Giuseppe Amato
* Fabio Carrara
* Fabrizio Falchi
* Claudio Gennaro
* Claudio Vairo

## Sitio Oficial

http://cnrpark.it/

## Descripción

CNRPark+EXT es un conjunto de datos diseñado para la detección visual de ocupación en estacionamientos.

Contiene aproximadamente 150,000 imágenes etiquetadas correspondientes a espacios de estacionamiento capturados por múltiples cámaras bajo diversas condiciones ambientales.

### Características principales

| Subconjunto | Cámaras | Imágenes |
| ----------- | ------- | -------- |
| CNRPark     | 2       | 12,584   |
| CNR-EXT     | 9       | ~138,000 |
| Total       | 11      | ~150,000 |

### Condiciones presentes en el dataset

* Soleado
* Nublado
* Lluvioso
* Sombras parciales
* Oclusiones parciales
* Distintos ángulos de visión
* Variaciones de iluminación



# Análisis Exploratorio de Datos (EDA)

## Estadísticas Generales

| Métrica             | Valor     |
| ------------------- | --------- |
| Total de imágenes   | Pendiente |
| Espacios ocupados   | Pendiente |
| Espacios libres     | Pendiente |
| Porcentaje ocupados | Pendiente |
| Porcentaje libres   | Pendiente |

## Distribución de Clases

La siguiente gráfica muestra la distribución de imágenes pertenecientes a cada categoría.

```text
outputs/figures/distribucion_clases.png
```

## Muestra Representativa de Imágenes

### Espacios Ocupados

Se presentan ejemplos de imágenes correspondientes a espacios ocupados.

### Espacios Libres

Se presentan ejemplos de imágenes correspondientes a espacios disponibles.

## Análisis de Resolución

| Métrica             | Valor     |
| ------------------- | --------- |
| Resolución mínima   | Pendiente |
| Resolución máxima   | Pendiente |
| Resolución promedio | Pendiente |



# Preparación de los Datos

## División del Dataset

Para garantizar una evaluación adecuada del modelo, el conjunto de datos se divide en:

| Conjunto      | Porcentaje |
| ------------- | ---------- |
| Entrenamiento | 70 %       |
| Validación    | 15 %       |
| Prueba        | 15 %       |

## Procesamiento Aplicado

* Organización de imágenes por clase.
* Normalización.
* Verificación de etiquetas.
* División aleatoria estratificada.
* Preparación para entrenamiento.



# Metodología

La metodología propuesta se compone de las siguientes etapas:

1. Adquisición del dataset.
2. Análisis exploratorio de datos.
3. Preprocesamiento de imágenes.
4. División del dataset.
5. Entrenamiento del modelo.
6. Evaluación del desempeño.
7. Análisis de resultados.

---

# Tecnologias Utilizadas

| Tecnologia          | Uso                          |
| ------------------- | ---------------------------- |
| Python 3.9+         | Desarrollo principal         |
| PyTorch             | Framework de Deep Learning   |
| TorchVision         | Transformaciones de imagen   |
| NumPy               | Procesamiento numerico       |
| Pandas              | Manipulacion de datos        |
| Scikit-learn        | Metricas y evaluacion        |
| OpenCV              | Procesamiento de video       |
| Matplotlib/Seaborn  | Visualizacion                |
| Google Colab        | Ejecucion de experimentos    |
| pytest              | Testing                      |


# Resultados

Los resultados finales serán documentados una vez concluido el entrenamiento y evaluación de los modelos.

## Métricas de Evaluación

| Métrica   | Valor     |
| --------- | --------- |
| Accuracy  | Pendiente |
| Precision | Pendiente |
| Recall    | Pendiente |
| F1-Score  | Pendiente |

## Matriz de Confusión

```text
           Predicción

           Libre  Ocupado
Libre        TN      FP
Ocupado      FN      TP
```


# Limitaciones

Entre las principales limitaciones del sistema se encuentran:

* Dependencia de la calidad de las cámaras.
* Cambios extremos de iluminación.
* Condiciones climáticas adversas.
* Oclusiones severas.
* Diferencias entre estacionamientos no presentes en el dataset.


# Trabajo Futuro

Las posibles extensiones del proyecto incluyen:

* Detección en tiempo real.
* Integración con aplicaciones móviles.
* Sistemas de guiado de estacionamiento.
* Uso de modelos de detección de objetos.
* Generación de mapas de ocupación en tiempo real.


# Referencias

Amato, G., Carrara, F., Falchi, F., Gennaro, C., & Vairo, C. (2017). Deep Learning for Decentralized Parking Lot Occupancy Detection. Expert Systems with Applications, 72, 327–334.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
