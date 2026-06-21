<<<<<<< HEAD
# chest_classification_System# chest_classification_System
# chest_classification_System
=======
# COVID-19 Chest Classification System

A data science project combining chest X-ray image classification (CNN) and
clinical vitals classification (Random Forest), served via a FastAPI backend.

## Project Structure

```
covid-chest-classifier/
├── data/
│   ├── clean_clinical_data.csv
│   └── COVID-19_Radiography_Dataset/   (download separately, see below)
├── models/                              (created after training)
│   ├── clinical_model.pkl
│   ├── clinical_scaler.pkl
│   ├── clinical_feature_meta.pkl
│   ├── covid_chest_xray_model.keras
│   └── class_indices.json
├── src/
│   ├── step1_data_loading_eda.py
│   ├── train_clinical.py
│   └── train_image_model.py
├── api/
│   ├── main.py
│   └── schemas.py
├── requirements.txt
└── README.md
```

## Setup (VS Code)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Prepare Clinical Data

Run the EDA/preprocessing script to generate `data/clean_clinical_data.csv`:
```bash
python src/step1_data_loading_eda.py
```

## Step 2: Download Image Dataset

The chest X-ray dataset (COVID-19 Radiography Database) is on Kaggle:
1. Get your API key from kaggle.com -> Account -> Create New API Token (`kaggle.json`)
2. Place it in `~/.kaggle/kaggle.json` (or `C:\Users\<you>\.kaggle\kaggle.json` on Windows)
3. Run:
   ```bash
   kaggle datasets download -d tawsifurrahman/covid19-radiography-database
   unzip covid19-radiography-database.zip -d data/
   ```
4. Make sure the resulting folder is `data/COVID-19_Radiography_Dataset/` with one
   subfolder per class (COVID, Normal, Lung_Opacity, Viral Pneumonia), each containing
   an `images/` subfolder.

## Step 3: Train Models

Train the clinical model (fast, runs on CPU):
```bash
python src/train_clinical.py
```

Train the image model (recommend GPU, will be slow on CPU):
```bash
python src/train_image_model.py
```

Both scripts save trained models/artifacts into `models/`.

## Step 4: Run the API

```bash
uvicorn api.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI where you can:
- POST to `/predict/clinical` with vitals/symptoms (JSON) -> returns COVID probability
- POST to `/predict/image` with a chest X-ray image file -> returns predicted class + confidence

The API will tell you (at `/`) which models are currently loaded, so you can run it
even if you've only trained one of the two models so far.

## Notes & Limitations

- The clinical dataset is heavily imbalanced (~1.4% positive), so the clinical model
  prioritizes recall via `class_weight="balanced"`, but precision remains low.
  Clinical vitals alone are not a reliable diagnostic signal.
- The image and clinical datasets are not paired by patient, so this project presents
  two complementary models rather than a single fused multimodal model.
- This is an educational/portfolio project, not a medical diagnostic tool.
>>>>>>> 8acb4153 (Initial commit)
