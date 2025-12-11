import pandas as pd
import numpy as np

def process_newborn_data(
    newborn_path: str,
    vaccine_schedule_path: str,
    drop_name: bool = True
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for newborn health monitoring and vaccination scheduling.

    Steps:
      1) Data cleaning
      2) Data integration (with Rwanda vaccine schedule)
      3) Data reduction (drop unnecessary columns)
      4) Data transformation (derived features)
      5) Data discretization (age groups, flags)
      6) Data augmentation (vaccine due/overdue features)
    """
    # -----------------------------
    # 1. Load and basic data cleaning
    # -----------------------------
    df = pd.read_csv(newborn_path)
    vacc = pd.read_csv(vaccine_schedule_path)

    # Ensure expected columns exist
    required_cols = [
        "babyid","name","gender","gestationalageweeks",
        "birthweightkg","birthlengthcm","birthheadcircumferencecm",
        "date","agedays","weightkg","lengthcm","headcircumferencecm",
        "temperature","heartratebpm","respiratoryratebpm","oxygensaturation",
        "feedingtype","feedingfrequencyperday","urineoutputcount","stoolcount",
        "jaundicelevelmgdl","apgarscore","immunizationsdone","reflexesnormal","risklevel"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in newborn file: {missing}")

    # Convert types
    numeric_cols = [
        "gestationalageweeks","birthweightkg","birthlengthcm","birthheadcircumferencecm",
        "agedays","weightkg","lengthcm","headcircumferencecm",
        "temperature","heartratebpm","respiratoryratebpm","oxygensaturation",
        "feedingfrequencyperday","urineoutputcount","stoolcount",
        "jaundicelevelmgdl","apgarscore"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing critical fields
    critical_cols = ["babyid","gender","agedays","weightkg","lengthcm","apgarscore"]
    df = df.dropna(subset=critical_cols).reset_index(drop=True)

    # -----------------------------
    # 2. Data integration: Vaccine schedule
    # -----------------------------
    # Rwanda vaccine schedule: id,Age,In_days,Vaccine,Dose,Notes [file:23]
    # We'll use In_days as the recommended age in days.
    vacc = vacc.copy()
    vacc["In_days"] = pd.to_numeric(vacc["In_days"], errors="coerce")

    # -----------------------------
    # 3. Data reduction: drop non-essential columns
    # -----------------------------
    if drop_name and "name" in df.columns:
        df = df.drop(columns=["name"])

    # -----------------------------
    # 4. Data transformation: derived features
    # -----------------------------
    # Age in months
    df["age_months"] = df["agedays"] / 30.4375

    # BMI-like metric
    df["bmi_like"] = df["weightkg"] / (df["lengthcm"] / 100.0) ** 2

    # Low birth weight & prematurity
    df["low_birth_weight"] = (df["birthweightkg"] < 2.5).astype(int)
    df["premature"] = (df["gestationalageweeks"] < 37).astype(int)

    # Fever
    df["fever_flag"] = (df["temperature"] >= 37.5).astype(int)

    # Simple global thresholds for vitals (you can refine by age group)
    df["oxygen_low_flag"] = (df["oxygensaturation"] < 95).astype(int)

    # Example age-group based thresholds (very simplified)
    # You can tune these using pediatric vital sign references.
    def tachycardia_threshold(age_days):
        if age_days <= 30:
            return 180
        elif age_days <= 365:
            return 170
        else:
            return 160

    def tachypnea_threshold(age_days):
        if age_days <= 30:
            return 60
        elif age_days <= 365:
            return 50
        else:
            return 40

    df["tachycardia_flag"] = df.apply(
        lambda row: int(row["heartratebpm"] > tachycardia_threshold(row["agedays"])),
        axis=1,
    )
    df["tachypnea_flag"] = df.apply(
        lambda row: int(row["respiratoryratebpm"] > tachypnea_threshold(row["agedays"])),
        axis=1,
    )

    # -----------------------------
    # 5. Data discretization: age groups
    # -----------------------------
    def age_group(days):
        if days <= 28:
            return "Neonate"
        elif days <= 365:
            return "Infant"
        elif days <= 3 * 365:
            return "Toddler"
        else:
            return "Preschool"

    df["age_group"] = df["agedays"].apply(age_group)


    # Pre-calc: list of vaccine milestones (ignore pre-birth)
    schedule = vacc[vacc["In_days"] >= 0][["In_days", "Vaccine"]].copy()

    def vacc_counts(row):
        age = row["agedays"]
        status = row["immunizationsdone"]
        due = (schedule["In_days"] <= age).sum()

        if status == "Yes":
            past_due = 0
        elif status == "No":
            past_due = due
        else:  # "Partial"
            past_due = int(round(due * 0.5))
        return pd.Series({"due_vaccines_count": due, "past_due_vaccines_count": past_due})

    vacc_features = df.apply(vacc_counts, axis=1)
    df = pd.concat([df, vacc_features], axis=1)

    # Example specific vaccine flags: MR1, MR2 (Measles-Rubella 1 and 2) [file:23]
    # MR1 at 270 days, MR2 at 450 days
    MR1_DAY = int(schedule[(schedule["Vaccine"] == "MR") & (schedule["In_days"] == 270)]["In_days"].min()) if any((schedule["Vaccine"] == "MR") & (schedule["In_days"] == 270)) else 270
    MR2_DAY = int(schedule[(schedule["Vaccine"] == "MR") & (schedule["In_days"] == 450)]["In_days"].min()) if any((schedule["Vaccine"] == "MR") & (schedule["In_days"] == 450)) else 450

    df["mr1_due"] = (df["agedays"] >= MR1_DAY).astype(int)
    df["mr2_due"] = (df["agedays"] >= MR2_DAY).astype(int)

    def overdue_flag(age, day, status):
        if age <= day:
            return 0
        if status == "Yes":
            return 0
        elif status == "No":
            return 1
        else:  # Partial
            return 1

    df["mr1_overdue"] = df.apply(
        lambda r: overdue_flag(r["agedays"], MR1_DAY, r["immunizationsdone"]), axis=1
    )
    df["mr2_overdue"] = df.apply(
        lambda r: overdue_flag(r["agedays"], MR2_DAY, r["immunizationsdone"]), axis=1
    )

    # Final: return processed dataframe
    return df
