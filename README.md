# IKIBONDO — Baby Health Risk Monitoring & Vaccination Scheduling (ML + Rules)

IKIBONDO is an AI-driven child health monitoring and vaccination scheduling support system for children aged 0–6 years, designed to assist Community Health Workers (CHWs) and health facilities with early risk screening and vaccine follow-up.  
This repository contains a preprocessing pipeline that engineers clinical + vaccination features and a model-training script that compares multiple classifiers under class imbalance using SMOTE. 

## What this project does

- Builds a processed dataset from raw newborn/child monitoring data and a Rwanda vaccination schedule file.   
- Trains and evaluates multiple ML models (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting) and selects the best model by Macro F1-score.  
- Applies SMOTE only to the training data (after preprocessing) to address imbalance in the risk classes (Low/Medium/High).  
- Saves the best SMOTE-trained pipeline (preprocessing + classifier) as a `.joblib` file for later inference/deployment. 

## Repository entry points

- `pipeline.py`: End-to-end feature engineering and vaccination feature creation (age groups, vitals flags, due/overdue counts, MR1/MR2 due/overdue). [file:171]  
- `preprocessing.py`: Calls `processNewbornData(...)` and exposes `processed_df` for training scripts to import. [file:173]  
- `HealthRiskModel_smote.py`: Trains models, prints class counts before/after SMOTE, reports metrics, and exports the best model pipeline. 

## Data files expected

By default, `preprocessing.py` loads: [file:173]  
- `datasets/health_monitoring_100k_rows.csv` [file:173]  
- `datasets/VaccineSchedule.csv` [file:173]

The pipeline expects key columns such as `babyid`, `gender`, `agedays`, anthropometrics/vitals, `immunizationsdone`, and `risklevel`. [file:171]

## How to run (training)

1) Install dependencies (recommended: use a virtual environment). [file:2]  
2) Ensure the dataset files exist at the paths used in `preprocessing.py`. [file:173]  
3) Run training:
