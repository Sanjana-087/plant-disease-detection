# 🌿 Plant Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered **plant leaf disease detection** system built for college research and demonstration. The project covers the full machine learning lifecycle: dataset preprocessing, training multiple deep learning architectures, rigorous evaluation (including cross-dataset domain shift analysis), Grad-CAM explainability, and a Streamlit web application for real-time inference.

The codebase is **modular**, **well-commented**, and designed so each component can be run independently from the command line or imported into Jupyter notebooks.

---

## Features

- **Multi-dataset support:** PlantVillage, PlantDoc, and custom Arecanut folder layouts
- **Four model architectures:** Custom CNN baseline, ResNet50, EfficientNetB0, EfficientNetB3
- **Two-phase transfer learning:** Frozen backbone → fine-tune top layers
- **Full metrics:** Accuracy, precision, recall, F1, confusion matrix, per-class CSV
- **Cross-dataset evaluation:** PlantVillage → PlantDoc with domain shift score
- **Grad-CAM:** Visual explanation of model decisions
- **Streamlit app:** Upload images, top-3 predictions, disease info cards, prediction history

---

## Architecture

```
Input Images (224×224×3)
         │
    ┌────▼────────────────────────────────────────┐
    │           PREPROCESSING PIPELINE            │
    │  Load → Resize → Normalize → Augment →      │
    │  Encode Labels → Split (70/15/15) →          │
    │  tf.data.Dataset (batched + prefetched)      │
    └────────────────┬────────────────────────────┘
                     │
         ┌───────────▼──────────────┐
         │      MODEL FACTORY       │
         │  ┌──────────────────┐    │
         │  │  CNN Baseline    │    │
         │  │  ResNet50 (TL)   │    │
         │  │  EfficientNetB0  │    │
         │  │  EfficientNetB3  │    │
         │  └──────────────────┘    │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │     TRAINING LOOP        │
         │  Phase 1: Frozen base    │
         │  Phase 2: Fine-tuning    │
         │  Callbacks: Checkpoint,  │
         │  EarlyStopping, LR decay │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │      EVALUATION          │
         │  In-dataset metrics      │
         │  Cross-dataset (domain   │
         │    shift analysis)       │
         │  Grad-CAM explainability │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │    STREAMLIT WEB APP     │
         │  Upload → Predict →      │
         │  Display + Grad-CAM      │
         │  + Prediction History    │
         └──────────────────────────┘
```

---

## Project Structure

```
plant_disease_detection/
│
├── dataset/
│   ├── plantvillage/          # Download manually
│   ├── plantdoc/              # Download manually
│   └── arecanut/              # Custom dataset
│
├── preprocessing/
│   ├── data_loader.py
│   ├── augmentation.py
│   └── split_dataset.py
│
├── models/
│   ├── cnn_baseline.py
│   ├── resnet50_model.py
│   ├── efficientnet_model.py
│   └── model_factory.py
│
├── training/
│   ├── config.py
│   ├── callbacks.py
│   └── train.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── cross_dataset_eval.py
│   └── grad_cam.py
│
├── app/
│   ├── streamlit_app.py
│   └── utils.py
│
├── utils/
│   ├── logger.py
│   ├── visualization.py
│   └── metrics.py
│
├── notebooks/                 # (optional exploration notebooks)
├── results/                   # Metrics, plots, logs (auto-generated)
├── saved_models/              # Trained .keras weights
│
├── requirements.txt
└── README.md
```

---

## Dataset Setup

### PlantVillage (~54,000 images, 38 classes)

1. Download from [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Extract so class folders live under:
   ```
   dataset/plantvillage/color/<Class_Name>/*.jpg
   ```
   (The data loader also supports flat `dataset/plantvillage/<Class_Name>/` layouts.)

### PlantDoc

1. Download from [GitHub — PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
2. Place under:
   ```
   dataset/plantdoc/
   ```
   Supports `train/`, `validation/`, `test/` splits or flat class folders.

### Arecanut (custom)

Organize as:
```
dataset/arecanut/<class_name>/*.jpg
```
Aim for **≥100 images per class** for reliable training.

---

## Installation

```bash
cd plant_disease_detection

# Create virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### GPU support (recommended for training)

```bash
# Linux / WSL with CUDA (TensorFlow 2.12+)
pip install tensorflow[and-cuda]

# Or use Google Colab / Kaggle Notebooks (free GPU)
```

---

## Usage

All commands assume your working directory is `plant_disease_detection/`.

### 1. Scan dataset (optional sanity check)

```bash
python -m preprocessing.data_loader --dataset plantvillage
```

### 2. Train a model

```bash
# CNN baseline (good for CPU / quick tests)
python -m training.train --model cnn_baseline --dataset plantvillage --epochs 50

# ResNet50 with two-phase transfer learning
python -m training.train --model resnet50 --dataset plantvillage --epochs 50 --batch-size 32

# EfficientNetB3
python -m training.train --model efficientnetb3 --dataset dataset/plantvillage --epochs 30
```

**Outputs:**
- `saved_models/{model}_best.keras` — best validation checkpoint
- `saved_models/{model}_final.keras` — final weights
- `results/{model}_history.json` — training curves
- `results/{model}_training_log.csv` — per-epoch metrics
- `results/label_mapping.json` — class index mapping

### 3. Evaluate on test set

```bash
python -m evaluation.evaluate \
  --model-path saved_models/resnet50_final.keras \
  --dataset plantvillage \
  --model-name resnet50
```

**Outputs:** confusion matrix PNG, per-class metrics CSV, evaluation summary JSON.

### 4. Cross-dataset evaluation (domain shift)

Requires models trained on PlantVillage and both datasets installed:

```bash
python -m evaluation.cross_dataset_eval \
  --model-path saved_models/resnet50_final.keras \
  --model-name resnet50
```

**Output:** `results/cross_dataset_results.csv` and printed comparison table.

### 5. Launch Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Select a model from `saved_models/`, upload a leaf image, and click **Detect Disease**.

---

## Model Comparison (fill after training)

| Model            | Dataset      | Accuracy | F1 (weighted) | Notes                    |
|------------------|-------------|----------|---------------|--------------------------|
| cnn_baseline     | PlantVillage | —        | —             | Fast baseline            |
| resnet50         | PlantVillage | —        | —             | Transfer learning        |
| efficientnetb0   | PlantVillage | —        | —             | Lightweight TL           |
| efficientnetb3   | PlantVillage | —        | —             | Higher capacity TL       |
| resnet50         | PlantDoc*    | —        | —             | Cross-dataset (mapped)   |

\*Cross-dataset row uses overlapping classes only; see `evaluation/cross_dataset_eval.py`.

> **Tip:** Quote only metrics from your own training runs during viva or demo — do not copy numbers from papers.

---

## Configuration

Central settings live in `training/config.py`:

| Parameter        | Default    |
|-----------------|------------|
| Image size      | 224×224    |
| Batch size      | 32         |
| Epochs          | 50         |
| Learning rate   | 1e-4       |
| Train/Val/Test  | 70/15/15   |
| Phase 1 epochs  | 10 (frozen backbone) |

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** — deep learning
- **scikit-learn** — metrics and stratified splits
- **OpenCV & Pillow** — image I/O and Grad-CAM overlays
- **Matplotlib & Seaborn** — plots
- **Streamlit** — web UI
- **pandas** — results tables
- **TensorBoard** — training visualization

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'tensorflow'` | Run `pip install -r requirements.txt` |
| Training very slow on CPU | Use Google Colab GPU or train `cnn_baseline` with fewer epochs |
| Streamlit: no models in dropdown | Train first; ensure `.keras` files are in `saved_models/` |
| Low cross-dataset accuracy | Expected domain shift; discuss in report |
| ResNet/EfficientNet poor accuracy | Ensure preprocessing matches model (handled automatically in pipelines) |

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute it for educational and research purposes. Dataset licenses (PlantVillage, PlantDoc) apply separately — check the original sources before redistribution.

---

## Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — primary training data
- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) — cross-dataset evaluation
- TensorFlow Model Garden — ResNet and EfficientNet implementations
