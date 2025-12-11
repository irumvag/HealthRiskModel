# IKIBONDO Baby Health Monitoring & Vaccination Data Preprocessing
## Project Overview
IKIBONDO is an AI-driven system for monitoring the health, growth, and vaccination schedules of children aged 0–6, using real and simulated data. This project provides all preprocessing steps needed for clean, analyzable data — from basic quality checks to integration with Rwanda’s national immunization calendar.

## Dataset Summary
Main health data:

`newborn_health_monitoring_with_risk.csv`
Includes baby demographics, growth, vitals, feeding records, milestones, risk labeling, and immunization completion.

Vaccination reference:

`rwanda_vaccination_schedule_days.csv`
Contains Rwanda’s immunization schedule mapped to age ranges (in days), doses, and target groups.

## Preprocessing Pipeline
### Data Cleaning:

Remove duplicates

Fill missing numeric values with median; categorical with mode

### Data Integration:

For every baby, intelligently assign scheduled vaccines using age windows from the Rwanda vaccination file (AgeStart_days, AgeEnd_days), including vaccine dose and group info.

### Data Reduction:

Keep only usefully analytic columns for modeling/analysis

### Data Transformation:

Min-Max scaling for continuous variables (weight, length, head circumference)

Label encoding for categories (gender)

### Data Discretization:

Convert continuous age (agedays) into categories (Infant, Toddler, Preschool)

### Data Augmentation:

Balance/expand dataset with synthetically modified (noisy) samples

## Code Requirements
Python 3.8+

pandas

numpy

matplotlib, seaborn (for visualization)

scikit-learn

Install required libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
How To Run
Place all datasets in your working directory.

Run the main script:

```bash
python model_ikibondo_v0.py
```
This will generate a cleaned, integrated, and transformed dataset as preprocessed_health_dataset.csv, plus visualizations and sample outputs.

Review all charts and tables (or use the notebook version for interactive visualization).

## Data Integration Example
Vaccines are matched using the code:

```python
for idx, row in vaccine_df.iterrows():
    mask = (
        (integrated_df['agedays'] >= row['AgeStart_days']) &
        (integrated_df['agedays'] <= row['AgeEnd_days'])
    )
    info = f"{row['Vaccine']} (Dose {row['Doses']}, Group: {row['TargetGroup']})"
    integrated_df.loc[mask, 'scheduled_vaccine'] = info
```
### Results Deliverables
Cleaned and integrated dataset

Analysis-ready CSV files

Bar charts and plots illustrating key preprocessing effects

Python/Jupyter code with documentation

## License
This repository and code are for educational project purposes under your institution’s coursework guidelines.