<![CDATA[<div align="center">

# 🛡️ Sentry Vision — AI Security System


           
**Real-time intelligent surveillance powered by deep learning**



           
[![CI Pipeline](https://github.com/Omar-Mahrous-am/Sentry_Vision-AI-Security-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/Omar-Mahrous-am/Sentry_Vision-AI-Security-System-/actions/workflows/ci.yml)
[![Deploy](https://github.com/Omar-Mahrous-am/Sentry_Vision-AI-Security-System-/actions/workflows/deploy.yml/badge.svg)](https://github.com/Omar-Mahrous-am/Sentry_Vision-AI-Security-System-/actions/workflows/deploy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)


           
---

An end-to-end AI-powered security surveillance system that integrates **weapon detection**, **fire & smoke detection**, and **license plate recognition (LPR)** into a unified platform with a FastAPI backend, real-time dashboard, and production-grade MLOps pipelines.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [AI Modules](#-ai-modules)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [CLI Scripts](#-cli-scripts)
- [CI/CD & MLOps](#-cicd--mlops)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

**Sentry Vision** is a modular, production-ready AI security system designed for real-time threat detection and automated alerting. It processes visual input (images, video streams, webcam feeds) through three specialized deep learning pipelines and exposes the results via a RESTful API with an integrated monitoring dashboard.

When a threat is detected — a weapon, fire, or a watchlisted license plate — the system triggers an **audible text-to-speech alert** in real time, enabling immediate response without human monitoring overhead.

---

## ✨ Key Features

| Category | Feature |
|---|---|
| 🔫 **Weapon Detection** | Real-time weapon identification using YOLOv8/YOLO11 object detection |
| 🔥 **Fire & Smoke Detection** | Binary classification via fine-tuned MobileNetV2 with live video/webcam support |
| 🚗 **License Plate Recognition** | Multi-stage pipeline: vehicle → plate → Arabic OCR → governorate classification |
| 🔊 **TTS Alerting** | Automated voice alerts on threat detection via `pyttsx3` |
| 📊 **Live Dashboard** | Real-time HTML dashboard served directly by the backend |
| 🗄️ **Watchlist Database** | SQLite-backed CRUD for stolen/wanted license plates |
| 🐳 **Containerized** | Docker & Docker Compose for one-command deployment |
| ⚙️ **MLOps Pipelines** | Automated CI, daily evaluation, weekly retraining, and CD via GitHub Actions |
| 🧪 **Tested** | Comprehensive test suite with `pytest` and coverage reporting |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │  Dashboard    │   │  Streamlit   │   │  REST Client │       │
│   │  (HTML)       │   │  Frontend    │   │  (curl/SDK)  │       │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │
└──────────┼──────────────────┼──────────────────┼────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (:8000)                       │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐       │
│   │ /predict     │  │/fire_predict│  │ /predict_weapon  │       │
│   │  (LPR)       │  │  (Fire)     │  │   (Weapon)       │       │
│   └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘       │
│          │                │                   │                  │
│   ┌──────▼──────┐  ┌──────▼──────┐  ┌────────▼─────────┐       │
│   │ LPR Pipeline │  │   Fire      │  │    Weapon         │       │
│   │ (3× YOLO)   │  │  Detector   │  │   Detector        │       │
│   │ + OCR Mapper │  │ (MobileNet) │  │  (YOLOv8)         │       │
│   └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘       │
│          │                │                   │                  │
│          └────────────────┼───────────────────┘                  │
│                           ▼                                      │
│                   ┌──────────────┐                                │
│                   │  TTS Engine  │  ← Audible alert on detection │
│                   └──────────────┘                                │
│                                                                  │
│   ┌──────────────────────────────────────┐                       │
│   │  SQLite DB (users.db)                │                       │
│   │  • Watchlisted plates (CRUD)         │                       │
│   │  • Prediction history                │                       │
│   └──────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Modules

### 1. Fire & Smoke Detection (`src/fire_detection/`)

| Detail | Value |
|---|---|
| **Architecture** | MobileNetV2 (transfer learning) |
| **Task** | Binary classification — Fire vs. No Fire |
| **Input** | 224×224 RGB images |
| **Normalization** | ImageNet mean/std |
| **Inference Modes** | Single image, video file, live webcam |
| **Model File** | `models/fire_model/fire_model.pth` (~9 MB) |

**Module structure:**
- `model.py` — MobileNetV2 backbone with custom binary classifier head
- `dataset.py` — Custom `Fire_Dataset` with video frame extraction
- `dataloader.py` — Train/val/test split data loaders
- `preprocessing.py` — Augmentation pipelines and normalization stats
- `train.py` — Full training loop with early stopping
- `evaluate.py` — Metrics computation, classification report, confusion matrix
- `inference.py` — `FireDetector` class for production inference
- `pipeline.py` — End-to-end inference pipeline
- `config.py` — Centralized hyperparameters and paths

### 2. Weapon Detection (`src/weapon_detect/`)

| Detail | Value |
|---|---|
| **Architecture** | YOLOv8 (Ultralytics) |
| **Task** | Object detection — weapon localization |
| **Model File** | `models/weapon_detection/best.pt` (~22 MB) |

**Module structure:**
- `predict.py` — `WeaponDetection` class with single-load model pattern
- `config.py` — Model path and device configuration

### 3. License Plate Recognition (`src/lpr/`)

A **three-stage cascade pipeline** optimized for Egyptian Arabic license plates:

```
Input Image → Vehicle Detection (YOLO11n) → Plate Detection (plate.pt) → OCR (best.pt) → Governorate Classification
```

| Stage | Model | Purpose |
|---|---|---|
| 1 | `yolo11n.pt` (~5.6 MB) | Vehicle localization |
| 2 | `plate.pt` (~5.4 MB) | License plate region detection |
| 3 | `best.pt` (~5.5 MB) | Arabic character & digit OCR |
| 4 | Rule-based mapper | Egyptian governorate classification |

**Module structure:**
- `pipeline.py` — `LPRPipeline` orchestrator with singleton pattern
- `detector.py` — `LPRDetector` wrapping three YOLO models
- `mapper.py` — Arabic character classification and governorate rules (27+ governorates)
- `config.py` — Model paths and confidence thresholds

### 4. Text-to-Speech Alerting (`src/tts/`)

- Non-blocking, thread-safe TTS engine using `pyttsx3`
- Speaks alerts directly on the server machine when threats are detected
- Background threading to avoid blocking API responses

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **AI / ML** | PyTorch 2.2, TorchVision, Ultralytics (YOLOv8/YOLO11), MobileNetV2 |
| **Backend** | FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| **Database** | SQLite |
| **Frontend** | HTML Dashboard, Streamlit |
| **TTS** | pyttsx3, gTTS |
| **Computer Vision** | OpenCV, Pillow |
| **ML Utilities** | scikit-learn, matplotlib, seaborn, NumPy |
| **DevOps** | Docker, Docker Compose, GitHub Actions, CircleCI |
| **Testing** | pytest, pytest-cov, pytest-asyncio, httpx |
| **Code Quality** | flake8, isort |

---

## 📁 Project Structure

```
Sentry_Vision-AI-Security-System/
│
├── backend/                        # FastAPI backend application
│   ├── system_backend.py           # Main app: endpoints, DB, model loading
│   ├── dashboard.html              # Backend-served monitoring dashboard
│   └── tts.py                      # Legacy TTS helper
│
├── frontend/
│   └── dashboard.html              # Streamlit-compatible frontend dashboard
│
├── src/                            # Core AI modules
│   ├── fire_detection/             # Fire & smoke detection module
│   │   ├── model.py                # MobileNetV2 architecture
│   │   ├── dataset.py              # Custom dataset (images + video frames)
│   │   ├── dataloader.py           # Data loading utilities
│   │   ├── preprocessing.py        # Augmentation & normalization
│   │   ├── train.py                # Training loop
│   │   ├── evaluate.py             # Metrics & confusion matrix
│   │   ├── inference.py            # Production inference engine
│   │   ├── pipeline.py             # End-to-end pipeline
│   │   └── config.py               # Hyperparameters & paths
│   │
│   ├── weapon_detect/              # Weapon detection module
│   │   ├── predict.py              # YOLO-based weapon detector
│   │   └── config.py               # Model path config
│   │
│   ├── lpr/                        # License Plate Recognition module
│   │   ├── pipeline.py             # Multi-stage LPR pipeline
│   │   ├── detector.py             # YOLO detector wrapper (3 models)
│   │   ├── mapper.py               # Arabic char & governorate classifier
│   │   └── config.py               # Model paths & thresholds
│   │
│   └── tts/                        # Text-to-Speech engine
│       ├── tts_engine.py           # Thread-safe pyttsx3 wrapper
│       └── config.py               # TTS configuration
│
├── scripts/                        # CLI scripts for training & inference
│   ├── run_system.py               # Fire detection inference (image/video/webcam)
│   ├── fire_train.py               # Fire model training script
│   ├── evaluate.py                 # Model evaluation script
│   ├── validate_dataset.py         # Dataset integrity validation
│   ├── run_tts.py                  # TTS testing utility
│   ├── prepare_data.py             # Data preparation script
│   └── train_plate.py              # Plate model training script
│
├── models/                         # Pre-trained model weights
│   ├── fire_model/
│   │   └── fire_model.pth          # MobileNetV2 fire detector (~9 MB)
│   ├── weapon_detection/
│   │   └── best.pt                 # YOLOv8 weapon detector (~22 MB)
│   └── License_plate/
│       ├── yolo11n.pt              # Vehicle detection (~5.6 MB)
│       ├── plate.pt                # Plate detection (~5.4 MB)
│       └── best.pt                 # Arabic OCR (~5.5 MB)
│
├── data/                           # Datasets (images, videos)
│   ├── img_data/
│   ├── video_data/
│   └── tts_output/
│
├── notebooks/                      # Jupyter notebooks (experimentation)
│   ├── fire-and-smoke-detection_training.ipynb
│   └── tts.ipynb
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Shared fixtures
│   ├── fire_detection/             # Fire module tests
│   ├── plate_recognition/          # LPR module tests
│   └── tts/                        # TTS module tests
│
├── .github/workflows/              # GitHub Actions CI/CD
│   ├── ci.yml                      # Lint → Test → Validate → Compile
│   ├── deploy.yml                  # Build → Push → Deploy (DockerHub)
│   ├── daily-eval.yml              # Daily model performance monitoring
│   └── retrain.yml                 # Weekly automated retraining
│
├── .circleci/
│   └── config.yml                  # CircleCI pipeline (test + Docker build)
│
├── Dockerfile                      # Production container definition
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Poetry project metadata
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── .dockerignore                   # Docker build exclusions
└── LICENSE                         # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.11+
- **pip** or **Poetry**
- **Git**
- System libraries: `libgl1`, `libglib2.0`, `libespeak1`, `ffmpeg` (Linux/Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/Omar-Mahrous-am/Sentry_Vision-AI-Security-System-.git
cd Sentry_Vision-AI-Security-System-
```

### 2. Set Up Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the template and edit with your values
cp .env.example .env
```

Key variables to configure:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment mode |
| `APP_PORT` | `8000` | Backend port |
| `DATABASE_URL` | `sqlite:///./users.db` | Database connection string |
| `FIRE_MODEL_PATH` | `models/fire_model/fire_model.pth` | Fire model weights |
| `WEAPON_MODEL_PATH` | `models/weapon_model/best.pt` | Weapon model weights |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### 5. Run the Backend

```bash
uvicorn backend.system_backend:app --host 0.0.0.0 --port 8000 --reload
```

The dashboard will be available at **http://localhost:8000**.

---

## 📡 API Reference

### Health & Dashboard

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the monitoring dashboard |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/redoc` | ReDoc API documentation |

### License Plate Recognition

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload an image for LPR. Returns detected plates, text, governorate, and watchlist match status |
| `POST` | `/save_predict` | Save a prediction result to the database |

**Example — LPR Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@car_photo.jpg"
```

**Response:**
```json
{
  "results": {
    "plates": [
      {
        "bbox": [120, 340, 480, 410],
        "text": "س ج ب 1234",
        "confidence": 0.92,
        "governorate": "Alexandria",
        "details": {
          "letters": ["س", "ج", "ب"],
          "digits": ["1", "2", "3", "4"]
        }
      }
    ]
  },
  "stolen_detected": false,
  "message": "No watchlisted plates detected"
}
```

### Fire Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/fire_predict` | Upload an image for fire/smoke detection |

**Example:**
```bash
curl -X POST http://localhost:8000/fire_predict \
  -F "image=@scene.jpg"
```

### Weapon Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict_weapon` | Upload an image for weapon detection |

**Example:**
```bash
curl -X POST http://localhost:8000/predict_weapon \
  -F "image=@surveillance.jpg"
```

### Watchlist Management (CRUD)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/plates` | List all watchlisted plates |
| `GET` | `/plates/{plate_number}` | Get a specific plate |
| `POST` | `/plates` | Add a plate to the watchlist |
| `PUT` | `/plates/{plate_number}` | Update a watchlisted plate |
| `DELETE` | `/plates/{plate_number}` | Remove a plate from the watchlist |
| `GET` | `/search/{plate_number}` | Search plates by partial match |

**Example — Add to Watchlist:**
```bash
curl -X POST http://localhost:8000/plates \
  -H "Content-Type: application/json" \
  -d '{"license_plate_number": "س ج ب 1234"}'
```

### Image Serving

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/Image/{image_name}` | Serve a stored image by filename |

---

## 💻 CLI Scripts

### Fire Detection Inference

Run detection on images, videos, or live webcam feeds:

```bash
# Single image
python scripts/run_system.py --mode image --source photo.jpg --weights models/fire_model/fire_model.pth

# Video file
python scripts/run_system.py --mode video --source clip.mp4 --weights models/fire_model/fire_model.pth --output result.mp4

# Live webcam
python scripts/run_system.py --mode webcam --weights models/fire_model/fire_model.pth
```

### Model Training

```bash
# Train the fire detection model
python scripts/fire_train.py --epochs 25 --batch-size 32 --lr 0.001
```

### Model Evaluation

```bash
# Evaluate on the test split
python scripts/evaluate.py --weights models/best_accuracy_model.pth

# With custom threshold and report output
python scripts/evaluate.py --weights models/best_accuracy_model.pth --threshold 0.6 --save-report eval/confusion_matrix.png
```

### Dataset Validation

```bash
python scripts/validate_dataset.py
```

---

## ⚙️ CI/CD & MLOps

The project implements a full MLOps lifecycle with **four GitHub Actions workflows** and a **CircleCI pipeline**:

### GitHub Actions

| Workflow | Trigger | Pipeline |
|---|---|---|
| **CI Pipeline** (`ci.yml`) | Push / PR to `main`, `develop` | Lint (flake8 + isort) → Unit Tests (pytest + coverage) → Dataset Validation → Compile & Import Check |
| **Deploy** (`deploy.yml`) | Merge to `main` or manual | Pre-flight Tests → Docker Build & Push to DockerHub → Production Deploy |
| **Daily Eval** (`daily-eval.yml`) | Cron (03:00 UTC daily) or manual | Load model → Run evaluation on test set → Upload metrics & confusion matrix |
| **Retraining** (`retrain.yml`) | Cron (06:00 UTC Sundays) or manual | Train new model → Upload checkpoint → Evaluate → Report results |

### CircleCI

| Job | Description |
|---|---|
| `test` | Run full test suite with coverage |
| `build-docker` | Build Docker image (runs on merge to main/develop) |

### Pipeline Flow

```
Code Push → Lint → Test → Validate Dataset → Compile Check
                                                    │
                                    (merge to main) │
                                                    ▼
                              Pre-flight → Docker Build → Push → Deploy

Daily (03:00 UTC): ──────► Evaluate Model → Upload Metrics

Weekly (Sunday 06:00 UTC): ──► Train → Evaluate → Upload Checkpoint
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f
```

This starts:
- **Backend** on port `8000` — FastAPI + AI inference
- **Frontend** on port `8501` — Streamlit dashboard

### Standalone Docker

```bash
# Build the image
docker build -t sentry-vision .

# Run the backend
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/users.db:/app/users.db \
  --name sentry-backend \
  sentry-vision
```

### Container Details

| Property | Value |
|---|---|
| **Base Image** | `python:3.11-slim` |
| **System Deps** | `libgl1`, `libglib2.0`, `libespeak1`, `ffmpeg` |
| **Exposed Ports** | `8000` (backend), `8501` (frontend) |
| **Entry Point** | `uvicorn backend.system_backend:app` |

---

## 🔧 Configuration

### Environment Variables

All configuration is managed via `.env` files. Copy the template to get started:

```bash
cp .env.example .env
```

See [`.env.example`](.env.example) for the full list of available variables, including:

- **App Settings** — host, port, debug mode
- **Database** — SQLAlchemy connection string
- **CORS** — allowed origins
- **Model Paths** — fire and weapon model locations
- **Upload Limits** — max file size (default: 5 MB)
- **Docker Hub** — credentials for CI/CD deployment
- **Frontend** — backend URL and Streamlit port
- **Logging** — log level configuration

### Model Configuration

Each AI module has its own `config.py` with tunable parameters:

| Module | Key Parameters |
|---|---|
| **Fire Detection** | `IMG_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `PATIENCE`, `THRESHOLD` |
| **Weapon Detection** | `WEAPON_MODEL_PATH`, `device` |
| **LPR** | `PLATE_CONF_THRESHOLD`, `OCR_CONF_THRESHOLD`, `DEVICE` |

---

## 🧪 Testing

Run the full test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# Run all tests
python -m pytest tests/ --verbose

# With coverage report
python -m pytest tests/ \
  --verbose \
  --cov=src \
  --cov=backend \
  --cov-report=term-missing \
  --cov-report=html:coverage-html
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures (mock models, test images)
├── fire_detection/          # Fire detection module tests
├── plate_recognition/       # LPR pipeline tests
└── tts/                     # TTS engine tests
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with descriptive messages
4. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open** a Pull Request against `main`

### Code Quality Standards

- **Linting:** `flake8` with max line length of 120
- **Imports:** Sorted with `isort`
- **Tests:** All new features must include tests
- **Documentation:** Update README and docstrings for API changes

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Omar Mahrous**

---

<div align="center">

*Built with ❤️ using PyTorch, FastAPI, and YOLO*

</div>
]]>
