# TIDE RF Pipeline

This repository contains a Random Forest pipeline for predicting whether startups receive follow-on funding under the TIDE 2.0 scheme. The project includes data preprocessing, model training, evaluation, and reporting.

---

## Project Structure
tide_rf_pipeline/
├── rf_pipeline.py # Main pipeline script
├── requirements.txt # Python dependencies
├── .gitignore # Ignore unnecessary files
├── model_metadata.json # Saved model metadata
├── rf_tide_model.pkl # Trained Random Forest model
└── rf_model_report.png # Visual report of model performance


---

## Features

- **Data Cleaning**: Converts messy input data into clean numeric and categorical features.
- **Feature Engineering**: Processes team size, funding amounts, TRL, sector, stage, and more.
- **Random Forest Model**: Uses `scikit-learn` RandomForestClassifier with hyperparameters tuned for TIDE 2.0 dataset.
- **Evaluation**: Provides train/test split, classification metrics, ROC-AUC, confusion matrix, and cross-validation.
- **Visualization**: Feature importance, model report, and performance metrics charts.
- **Model Saving**: Saves trained model and metadata for future predictions.

---

## Installation      

1. Clone the repository:

```bash
git clone https://github.com/devansupant-2006/tide_rf_pipeline.git
cd tide_rf_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


