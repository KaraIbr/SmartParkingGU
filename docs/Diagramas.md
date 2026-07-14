# Diagramas del Proyecto SmartParkingGU

Diagramas Mermaid que explican la arquitectura, flujo de datos y operacion del sistema de deteccion de ocupacion de estacionamiento.

---

## 1. Flujo General del Proyecto

```mermaid
flowchart TD
    subgraph FASE1["FASE 1: Adquisicion de Datos"]
        A1[("Google Drive\nCNRPark-Patches\n~12K imagenes")]
        A2[("Videos de Vigilancia\nEstacionamiento\n2A, 2B, 2C")]
        A3["Montar en Google Colab"]
        A4["Extraer frames con\nOpenCV (1 FPS)"]
    end

    subgraph FASE2["FASE 2: Analisis Exploratorio"]
        B1["01_dataset_overview.ipynb"]
        B2["Distribucion de clases"]
        B3["Analisis de resolucion"]
        B4["Muestras representativas"]
        B5["Analisis HOG"]
    end

    subgraph FASE3["FASE 3: Preprocesamiento"]
        C1["ParkingDataset\nscripts/data.py"]
        C2["Normalizacion\nImageNet mean/std"]
        C3["Augmentation\nFlip + Rotation"]
        C4["Division estratificada\n70% / 15% / 15%"]
        C5["DataLoaders\nbatch=32"]
    end

    subgraph FASE4["FASE 4: Entrenamiento Base"]
        D1["SmartParkingCNN\nscripts/models.py"]
        D2["Adam optimizer\nlr=1e-3"]
        D3["CrossEntropyLoss\ncon pesos de clase"]
        D4["EarlyStopping\npatience=10"]
        D5["Checkpoints\nexperiments/base_model/"]
    end

    subgraph FASE5["FASE 5: Fine-Tuning"]
        E1["Grid Search\n16 experimentos"]
        E2["Random Search\n15 experimentos"]
        E3["Metricas: F1, Accuracy,\nPrecision, Recall"]
        E4["Mejor configuracion"]
        E5["Modelo final\nexperiments/best_model/"]
    end

    subgraph FASE6["FASE 6: Evaluacion y Exportacion"]
        F1["Matriz de Confusion"]
        F2["Curva ROC + AUC"]
        F3["Curva Precision-Recall"]
        F4["Comparacion de modelos"]
        F5["Serializacion\nmodels/final_model.pt"]
    end

    A1 --> A3
    A2 --> A4
    A3 --> B1
    A4 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5

    B2 --> C1
    B3 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5

    C5 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5

    D5 --> E1
    D5 --> E2
    E1 --> E3
    E2 --> E3
    E3 --> E4
    E4 --> E5

    E5 --> F1
    E5 --> F2
    E5 --> F3
    F1 --> F4
    F2 --> F4
    F3 --> F4
    F4 --> F5

    style FASE1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style FASE2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style FASE3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style FASE4 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style FASE5 fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style FASE6 fill:#e0f7fa,stroke:#00838f,stroke-width:2px
```

---

## 2. Pipeline de Datos

```mermaid
flowchart LR
    subgraph FUENTES["Fuentes de Datos"]
        D1[("CNRPark Dataset\n~12,584 imagenes\n150x150 px")]
        D2[("Videos MP4\n3 camaras\n20 FPS originales")]
    end

    subgraph EXTRACCION["Extraccion"]
        E1["extract_frames.py\nOpenCV VideoCapture"]
        E2["1 frame por segundo"]
        E3["Resize a 224x224"]
        E4["180 frames totales"]
        E5["frames_metadata.csv"]
    end

    subgraph ANALISIS["Analisis CV"]
        AV1["Deteccion de bordes\nCanny"]
        AV2["Deteccion de lineas\nHough"]
        AV3["Deteccion de contornos\nAdaptative Threshold"]
        AV4["Histogramas de brillo"]
    end

    subgraph DATASET["ParkingDataset"]
        DS1["Clases binarias:\nOcupado / Libre"]
        DS2["Normalizacion\nmean=[0.485,0.456,0.406]\nstd=[0.229,0.224,0.225]"]
        DS3["Augmentation\nRandomHorizontalFlip\nRandomRotation(10)"]
    end

    subgraph DIVISION["Division Estratificada"]
        TR["Entrenamiento\n70%\n~8,800 imagenes"]
        VA["Validacion\n15%\n~1,888 imagenes"]
        TE["Prueba\n15%\n~1,888 imagenes"]
    end

    subgraph DATALOADERS["DataLoaders"]
        DL["batch_size=32\nshuffle=True\nnum_workers=2"]
    end

    D1 --> DS1
    D2 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    E3 --> AV1
    E3 --> AV2
    E3 --> AV3
    E3 --> AV4

    DS1 --> DS2
    DS2 --> DS3
    DS3 --> TR
    DS3 --> VA
    DS3 --> TE

    TR --> DL
    VA --> DL
    TE --> DL

    style FUENTES fill:#e3f2fd,stroke:#1565c0
    style EXTRACCION fill:#fff3e0,stroke:#ef6c00
    style ANALISIS fill:#f3e5f5,stroke:#7b1fa2
    style DATASET fill:#e8f5e9,stroke:#2e7d32
    style DIVISION fill:#fce4ec,stroke:#c62828
    style DATALOADERS fill:#e0f7fa,stroke:#00838f
```

---

## 3. Arquitectura SmartParkingCNN

```mermaid
flowchart TD
    INPUT["Entrada\n[B, 3, 150, 150]\nRGB normalizado"]

    subgraph BLOQUE1["Bloque Conv 1"]
        C1A["Conv2d(3, 32, kernel=3, padding=1)"]
        C1B["BatchNorm2d(32)"]
        C1C["ReLU"]
        C1D["MaxPool2d(2)"]
        C1S["Salida: [B, 32, 75, 75]"]
    end

    subgraph BLOQUE2["Bloque Conv 2"]
        C2A["Conv2d(32, 64, kernel=3, padding=1)"]
        C2B["BatchNorm2d(64)"]
        C2C["ReLU"]
        C2D["MaxPool2d(2)"]
        C2S["Salida: [B, 64, 37, 37]"]
    end

    subgraph BLOQUE3["Bloque Conv 3"]
        C3A["Conv2d(64, 128, kernel=3, padding=1)"]
        C3B["BatchNorm2d(128)"]
        C3C["ReLU"]
        C3D["MaxPool2d(2)"]
        C3S["Salida: [B, 128, 18, 18]"]
    end

    subgraph BLOQUE4["Bloque Conv 4"]
        C4A["Conv2d(128, 256, kernel=3, padding=1)"]
        C4B["BatchNorm2d(256)"]
        C4C["ReLU"]
        C4D["MaxPool2d(2)"]
        C4S["Salida: [B, 256, 9, 9]"]
    end

    FLATTEN["Flatten\n256 x 9 x 9 = 20,736"]

    subgraph FC["Capas Fully Connected"]
        FC1["Linear(20736, 512)"]
        FC1R["ReLU"]
        FC1D["Dropout(0.5)"]
        FC2["Linear(512, 2)"]
    end

    SOFTMAX["Softmax"]

    subgraph SALIDA["Salida"]
        S1["Clase 0: Ocupado"]
        S2["Clase 1: Libre"]
    end

    INPUT --> C1A
    C1A --> C1B --> C1C --> C1D --> C1S
    C1S --> C2A
    C2A --> C2B --> C2C --> C2D --> C2S
    C2S --> C3A
    C3A --> C3B --> C3C --> C3D --> C3S
    C3S --> C4A
    C4A --> C4B --> C4C --> C4D --> C4S
    C4S --> FLATTEN
    FLATTEN --> FC1
    FC1 --> FC1R --> FC1D --> FC2
    FC2 --> SOFTMAX
    SOFTMAX --> S1
    SOFTMAX --> S2

    style INPUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style BLOQUE1 fill:#e8f5e9,stroke:#2e7d32
    style BLOQUE2 fill:#f1f8e9,stroke:#558b2f
    style BLOQUE3 fill:#fff8e1,stroke:#f9a825
    style BLOQUE4 fill:#fff3e0,stroke:#ef6c00
    style FLATTEN fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style FC fill:#f3e5f5,stroke:#7b1fa2
    style SOFTMAX fill:#e0f7fa,stroke:#00838f,stroke-width:2px
    style SALIDA fill:#e0f7fa,stroke:#00838f
```

---

## 4. Modulos del Codigo (Relaciones)

```mermaid
flowchart TB
    subgraph NOTEBOOKS["Notebooks (Google Colab)"]
        N1["01_dataset_overview.ipynb"]
        N2["02_base_model_cnrpark.ipynb"]
        N3["03_finetune_cnrpark.ipynb"]
        N4["COLAB_EJECUCION.ipynb"]
    end

    subgraph SCRIPTS["Paquete scripts/"]
        direction TB
        M_DATA["data.py\nParkingDataset\nget_default_transforms"]
        M_MODEL["models.py\nSmartParkingCNN\nget_model_info"]
        M_TRAIN["train.py\ntrain_model\nEarlyStopping\nload_checkpoint"]
        M_EVAL["evaluate.py\nevaluate_model\nplot_confusion_matrix\nplot_roc_curve\nplot_pr_curve\nplot_training_curves\ncompare_models"]
        M_HYP["hyperparameter_search.py\ngrid_search\nrandom_search\nsave_results_to_csv\nfind_best_config"]
        M_INF["inference.py\nSmartParkingPredictor\npredict\npredict_batch"]
        M_EXT["extract_frames.py\nextract_frames\nprocess_video_dir"]
        M_VID["analyze_videos.py\nanalyze_video"]
    end

    subgraph TESTS["Tests (pytest)"]
        T1["test_models.py"]
        T2["test_train.py"]
        T3["test_evaluate.py"]
        T4["test_data.py"]
    end

    subgraph DATOS["Datos y Modelos"]
        DATA[("data/\nCNRPark\n~12K imagenes")]
        VIDS[("Video/\n3 archivos MP4")]
        EXP[("experiments/\nresultados CSV\ncheckpoints .pt")]
        MOD[("models/\nfinal_model.pt")]
    end

    N1 --> M_DATA
    N2 --> M_DATA
    N2 --> M_MODEL
    N2 --> M_TRAIN
    N2 --> M_EVAL
    N3 --> M_DATA
    N3 --> M_MODEL
    N3 --> M_TRAIN
    N3 --> M_EVAL
    N3 --> M_HYP
    N4 --> M_DATA
    N4 --> M_MODEL
    N4 --> M_TRAIN

    M_DATA --> M_TRAIN
    M_MODEL --> M_TRAIN
    M_TRAIN --> M_EVAL
    M_TRAIN --> EXP
    M_HYP --> M_TRAIN
    M_HYP --> EXP
    M_EVAL --> EXP

    M_EXT --> VIDS
    M_VID --> VIDS

    M_DATA --> DATA

    M_INF --> MOD

    T1 --> M_MODEL
    T2 --> M_TRAIN
    T3 --> M_EVAL
    T4 --> M_DATA

    style NOTEBOOKS fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style SCRIPTS fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style TESTS fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style DATOS fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

---

## 5. Flujo de Inferencia (Prediccion)

```mermaid
sequenceDiagram
    actor U as Usuario
    participant I as SmartParkingPredictor
    participant P as Preprocesamiento
    participant C as SmartParkingCNN
    participant S as Softmax
    participant R as Resultado

    U->>I: predict("imagen_estacionamiento.jpg")
    activate I
    I->>I: Cargar modelo .pt\n(una sola vez al iniciar)
    I->>P: imagen PIL
    activate P
    P->>P: Resize(150 x 150)
    P->>P: ToTensor()
    P->>P: Normalize(ImageNet\nmean/std)
    deactivate P
    P->>C: tensor [1, 3, 150, 150]
    activate C
    C->>C: Conv Block 1\n3 -> 32 canales
    C->>C: Conv Block 2\n32 -> 64 canales
    C->>C: Conv Block 3\n64 -> 128 canales
    C->>C: Conv Block 4\n128 -> 256 canales
    C->>C: Flatten + FC\n20736 -> 512 -> 2
    deactivate C
    C->>S: logits [1, 2]
    activate S
    S->>S: exp(logit) / sum(exp)
    deactivate S
    S->>R: probabilidades [0.XX, 0.YY]
    activate R
    R->>R: clase = argmax
    R->>R: confianza = max(prob)
    deactivate R
    R-->>I: {clase: "Ocupado", confianza: 0.93}
    I-->>U: Prediccion completa
    deactivate I
```

---

## 6. Entorno de Ejecucion

```mermaid
flowchart LR
    subgraph COLAB["Google Colab"]
        GPU["GPU T4\n16 GB VRAM"]
        PY["Python 3.10"]
        TORCH["PyTorch 2.0\ntorchvision"]
        CKPT["Checkpoints\nExperimentos"]
    end

    subgraph TUNNEL["Conexion"]
        LT["localtunnel\npuerto remoto"]
        JV["Jupyter Server\nremoto"]
    end

    subgraph LOCAL["Desarrollo Local"]
        VSC["VS Code\nEditor principal"]
        PYL["Python 3.9+\nlocal"]
        PYT["pytest\nTests unitarios"]
        RUFF["ruff\nLinter"]
        MY["mypy\nType checker"]
    end

    subgraph GITHUB["GitHub"]
        REPO["Repositorio\nKaraIbr/SmartParkingGU"]
        CI["GitHub Actions\nCI Pipeline"]
        PYTEST_CI["pytest + coverage\nPython 3.9/3.10/3.11"]
    end

    subgraph FLUJO_CI["Flujo CI"]
        PUSH["git push\nmain"]
        TRIGGER["Trigger:\npush / PR"]
        MATRIX["Matrix Build\nUbuntu + Py 3.9/3.10/3.11"]
        TEST_CI["Tests + Coverage"]
        BADGE["Badge CI ✓/✗"]
    end

    GPU --> JV
    JV --> LT
    LT --> VSC

    VSC --> REPO
    REPO --> CI
    CI --> PYTEST_CI
    PYTEST_CI --> BADGE

    PUSH --> TRIGGER
    TRIGGER --> MATRIX
    MATRIX --> TEST_CI
    TEST_CI --> BADGE

    style COLAB fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style TUNNEL fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style LOCAL fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style GITHUB fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style FLUJO_CI fill:#fce4ec,stroke:#c62828,stroke-width:2px
```
