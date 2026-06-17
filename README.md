# TIDE RF Pipeline

This repository contains a Random Forest pipeline for predicting whether startups receive follow-on funding under the TIDE 2.0 scheme. The project includes data preprocessing, model training, evaluation, and reporting.

---

## Project Structure

```text
tide_rf_pipeline/
├── rf_pipeline.py        # Main pipeline script
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignore unnecessary files
├── model_metadata.json   # Saved model metadata
├── rf_tide_model.pkl     # Trained Random Forest model
└── rf_model_report.png   # Visual report of model performance
```



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
python rf_pipeline.py
```

## Model Interpretability & SHAP Analysis
To move beyond "black-box" predictions, this repository includes a dedicated `shap_analysis.py` script. By utilizing **SHAP (SHapley Additive exPlanations)**, we can quantify the contribution of each feature to the model's decision-making process for startups under the TIDE 2.0 scheme.


---

## Visual Insights
Our analysis produces the following diagnostic visualizations:
## Running the Analysis
After training the model using `rf_pipeline.py`, you can generate the SHAP diagnostic plots by running:
```bash
python shap_analysis.py
```

| Visualization | Description |
| :--- | :--- |
| [Summary Bar Plot](rf_model_report.png) | Ranks features by global importance based on the mean absolute SHAP value. |
| [Beeswarm Plot](shap_summary_beeswarm.png) | Shows how the magnitude and direction of feature values correlate with the model's funding prediction. |
| [Dependence Grid](shap_dependence_grid.png) | Visualizes the non-linear relationship between top features and their impact on the model output. |
| [Waterfall Examples](shap_waterfall_examples.png) | Explains the specific decision path for individual True Positive and True Negative startup predictions. |

## Key Findings
* **Top Predictors:** `Amount Sanctioned` and `Team Size` emerged as the most significant drivers of the model's predictions.
* **Feature Impact:** The beeswarm plot demonstrates that higher values for certain features (red dots) push the model's output significantly toward a "Funded" classification, providing clear evidence for the decision-making patterns the Random Forest has learned.


