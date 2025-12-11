from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib, os
import numpy as np
from preprocessing import processed_df

# 1. Build label and features
df_v = processed_df.copy()

def compute_future_overdue_risk(row):
    # You can tweak this logic later if needed
    if row["agedays"] > 450:
        return 0
    if row["past_due_vaccines_count"] > 0:
        return 1
    if row["immunizationsdone"] in ["No", "Partial"]:
        return 1
    return 0

df_v["future_overdue_risk"] = df_v.apply(compute_future_overdue_risk, axis=1)
df_v = df_v.dropna(subset=["future_overdue_risk"])
y_future = df_v["future_overdue_risk"].astype(int)

print("future_overdue_risk value counts:")
print(y_future.value_counts())

numeric_vacc = [
    "agedays","age_months",
    "gestationalageweeks",
    "due_vaccines_count"
]

binary_vacc = [
    "premature","low_birth_weight"
]

categorical_vacc = [
    "gender","age_group","immunizationsdone","feedingtype"
]

X_future = df_v[numeric_vacc + binary_vacc + categorical_vacc]

# 2. Split
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_future, y_future, test_size=0.2, random_state=42, stratify=y_future
)

print("Train class counts:", np.bincount(y_train_v))

# 3. Preprocessing
num_trans_v = Pipeline(steps=[("scaler", StandardScaler())])
cat_trans_v = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preproc_v = ColumnTransformer(
    transformers=[
        ("num", num_trans_v, numeric_vacc + binary_vacc),
        ("cat", cat_trans_v, categorical_vacc),
    ]
)

# 4. Models (binary)
models_v = {
    "log_reg": LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "decision_tree": DecisionTreeClassifier(max_depth=None, random_state=42, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    #"gbm": GradientBoostingClassifier(random_state=42),
}

results_v = {}
best_name_v, best_f1_v, best_pipe_v = None, -1.0, None

for name, clf in models_v.items():
    print(f"\n=== Future vaccination risk model: {name} ===")

    # If for some reason train set has only one class, skip LR to avoid error
    if len(np.unique(y_train_v)) < 2 and isinstance(clf, LogisticRegression):
        print("Skipping log_reg – only one class present in y_train_v.")
        continue

    pipe = Pipeline(steps=[("preprocess", preproc_v), ("clf", clf)])
    pipe.fit(X_train_v, y_train_v)
    y_pred_v = pipe.predict(X_test_v)

    print(classification_report(y_test_v, y_pred_v, digits=3))
    print(confusion_matrix(y_test_v, y_pred_v))

    mf1 = f1_score(y_test_v, y_pred_v, average="macro")
    results_v[name] = mf1
    print(f"Macro F1: {mf1:.3f}")

    if mf1 > best_f1_v:
        best_f1_v, best_name_v, best_pipe_v = mf1, name, pipe

print("\nFuture vaccination model macro F1:", results_v)
print("Best future vaccination model:", best_name_v, "with macro F1 =", best_f1_v)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_pipe_v, "saved_models/FutureVaccinationRiskModel_best.joblib")
print("Saved best future vaccination model to saved_models/FutureVaccinationRiskModel_best.joblib")
