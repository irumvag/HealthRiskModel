# generate_visualizations.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

from preprocessing import processed_df  # <- produced by pipeline.process_newborn_data()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_class_distribution(y, class_names, outpath, title):
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    ax.bar(class_names, counts.values, color=colors, edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel("Number of records")
    ax.grid(axis="y", alpha=0.25)

    total = counts.sum()
    for i, c in enumerate(counts.values):
        ax.text(i, c + (0.01 * total), f"{c}\n({c/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return counts.to_dict()


def plot_smote_before_after(y_before, y_after, class_names, outpath):
    before_counts = pd.Series(y_before).value_counts().sort_index()
    after_counts = pd.Series(y_after).value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    ax1.bar(class_names, before_counts.values, color=colors, edgecolor="black", alpha=0.75)
    ax1.set_title("Before SMOTE (training split)")
    ax1.set_ylabel("Number of records")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(class_names, after_counts.values, color=colors, edgecolor="black", alpha=0.75)
    ax2.set_title("After SMOTE (training split)")
    ax2.set_ylabel("Number of records")
    ax2.grid(axis="y", alpha=0.25)

    plt.suptitle("Class Distribution: Impact of SMOTE")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return before_counts.to_dict(), after_counts.to_dict()


def plot_confusion_matrix_dual(y_true, y_pred, class_names, outpath):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Count"}, ax=ax1)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Proportion"}, ax=ax2)
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cm, cm_norm


def plot_feature_importance_dt(dt_model, feature_names, outpath, top_n=15):
    importances = dt_model.feature_importances_
    fi = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(top_n))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(fi["Feature"][::-1], fi["Importance"][::-1],
            color="steelblue", edgecolor="navy", alpha=0.75)
    ax.set_title("Top Feature Importance (Decision Tree)")
    ax.set_xlabel("Importance score")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for y, v in enumerate(fi["Importance"][::-1]):
        ax.text(v + fi["Importance"].max() * 0.01, y, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fi


def plot_model_comparison(results_df, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
    titles = ["Accuracy", "Macro F1-score", "Macro Precision", "Macro Recall"]

    for idx, (m, t) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(results_df.index, results_df[m], color="steelblue", edgecolor="navy", alpha=0.75)
        ax.set_ylim(0, 1)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.tick_params(axis="x", rotation=25)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Machine Learning Model Performance Comparison", y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # Output folder
    outdir = "figures"
    ensure_dir(outdir)

    # =========================
    # 1) Same target mapping as your training script
    # =========================
    df = processed_df.copy()
    df = df.dropna(subset=["risklevel"])
    y = df["risklevel"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)

    class_names = ["Low Risk", "Medium Risk", "High Risk"]

    # =========================
    # 2) Same feature columns as your training script
    # =========================
    numeric_features = [
        "gestationalageweeks", "agedays", "age_months",
        "birthweightkg", "birthlengthcm", "birthheadcircumferencecm",
        "weightkg", "lengthcm", "headcircumferencecm", "bmi_like",
        "temperature", "heartratebpm", "respiratoryratebpm", "oxygensaturation",
        "feedingfrequencyperday", "urineoutputcount", "stoolcount",
        "jaundicelevelmgdl", "apgarscore",
        "due_vaccines_count", "past_due_vaccines_count"
    ]

    binary_features = [
        "low_birth_weight", "premature",
        "fever_flag", "tachycardia_flag", "tachypnea_flag", "oxygen_low_flag",
        "mr1_due", "mr1_overdue", "mr2_due", "mr2_overdue"
    ]

    categorical_features = ["gender", "feedingtype", "reflexesnormal", "immunizationsdone", "age_group"]

    feature_cols = numeric_features + binary_features + categorical_features
    X = df[feature_cols]

    # =========================
    # 3) Train/test split (same strategy)
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Figure: class distribution before SMOTE (training split)
    plot_class_distribution(
        y_train, class_names,
        os.path.join(outdir, "class_distribution_before_smote.png"),
        "Class distribution (training split, before SMOTE)"
    )

    # =========================
    # 4) Preprocessing (fit on train only)
    # =========================
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features + binary_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get expanded feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # =========================
    # 5) SMOTE on training only
    # =========================
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    plot_smote_before_after(
        y_train, y_train_bal, class_names,
        os.path.join(outdir, "smote_class_distribution.png")
    )

    # =========================
    # 6) Train models and store metrics
    # =========================
    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "k-NN": KNeighborsClassifier(n_neighbors=7),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    results = []
    best_dt = None
    best_dt_f1 = -1

    for name, clf in models.items():
        clf.fit(X_train_bal, y_train_bal)
        y_pred = clf.predict(X_test_proc)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }
        results.append(row)

        # Keep DT for feature importance + confusion matrix (or keep best overall if you want)
        if name == "DecisionTree" and row["macro_f1"] > best_dt_f1:
            best_dt_f1 = row["macro_f1"]
            best_dt = clf

    results_df = pd.DataFrame(results).set_index("model").sort_values("macro_f1", ascending=False)
    results_df.to_csv(os.path.join(outdir, "model_metrics_table.csv"), index=True)

    # Model comparison chart
    plot_model_comparison(results_df, os.path.join(outdir, "model_comparison.png"))

    # =========================
    # 7) Confusion matrix for Decision Tree
    # =========================
    dt_pred = best_dt.predict(X_test_proc)
    plot_confusion_matrix_dual(
        y_test, dt_pred, class_names,
        os.path.join(outdir, "confusion_matrix_multiclass.png")
    )

    # =========================
    # 8) Feature importance for Decision Tree
    # =========================
    plot_feature_importance_dt(
        best_dt, feature_names,
        os.path.join(outdir, "feature_importance_decision_tree.png"),
        top_n=15
    )

    print("\nSaved figures to:", os.path.abspath(outdir))
    print("Saved metrics table:", os.path.join(outdir, "model_metrics_table.csv"))
    print("\nTop models by Macro F1:\n", results_df[["macro_f1", "accuracy"]].head(5))


if __name__ == "__main__":
    main()
