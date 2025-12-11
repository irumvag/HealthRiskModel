import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from preprocessing import processed_df
import joblib
import os

# =========================================
# 1. Define features and target
# =========================================

df = processed_df.copy()
df = df.dropna(subset=["risklevel"])

y_health = df["risklevel"].map({"Low": 0, "Medium": 1, "High": 2})

numeric_features = [
    "gestationalageweeks","agedays","age_months",
    "birthweightkg","birthlengthcm","birthheadcircumferencecm",
    "weightkg","lengthcm","headcircumferencecm","bmi_like",
    "temperature","heartratebpm","respiratoryratebpm","oxygensaturation",
    "feedingfrequencyperday","urineoutputcount","stoolcount",
    "jaundicelevelmgdl","apgarscore",
    "due_vaccines_count","past_due_vaccines_count"
]

binary_features = [
    "low_birth_weight","premature",
    "fever_flag","tachycardia_flag","tachypnea_flag","oxygen_low_flag",
    "mr1_due","mr1_overdue","mr2_due","mr2_overdue"
]

categorical_features = [
    "gender","feedingtype","reflexesnormal","immunizationsdone","age_group"
]

feature_cols = numeric_features + binary_features + categorical_features
X_health = df[feature_cols]

# =========================================
# 2. Train/validation split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_health, y_health, test_size=0.2, random_state=42, stratify=y_health
)

# =========================================
# 3. Preprocessing
# =========================================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features + binary_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =========================================
# 4. Define the 5 models (no SMOTE, with class_weight)
# =========================================
models = {
    "log_reg": LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "decision_tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "gbm": GradientBoostingClassifier(
        random_state=42
    ),
}

# 5. Train and evaluate models

results = {}
best_model_name = None
best_macro_f1 = -1.0
best_pipeline = None

for name, clf in models.items():
    print(f"\n=== Training {name} ===")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    results[name] = macro_f1
    print(f"Macro F1: {macro_f1:.3f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_model_name = name
        best_pipeline = pipe

print("\nModel macro F1 scores:", results)
print("Best model:", best_model_name, "with macro F1 =", best_macro_f1)

# 6. Save best HealthRiskModel
os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_pipeline, "saved_models/HealthRiskModel_best.joblib")
print("Saved best HealthRiskModel to saved_models/HealthRiskModel_best.joblib")
