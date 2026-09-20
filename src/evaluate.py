"""
Evaluation and visualization module for the Heart Failure Detection System.
Provides functions to compute classification metrics, generate comparative ROC curves,
confusion matrices, feature importance charts, and comprehensive EDA plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)

# Apply consistent aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes key classification metrics with emphasis on Recall and F1-score.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["True Negatives"] = tn
    metrics["False Positives"] = fp
    metrics["False Negatives"] = fn
    metrics["True Positives"] = tp

    return metrics


def plot_eda_figures(df, reports_dir):
    """
    Generates and saves essential exploratory data analysis visualizations:
    1. Target class distribution
    2. Numerical features distributions (with KDE)
    3. Key categorical features stratified by Heart Disease target
    4. Correlation matrix of numerical variables
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_COL].value_counts()
    percentages = df[TARGET_COL].value_counts(normalize=True) * 100
    bars = ax.bar(["Normal (0)", "Heart Disease (1)"], counts, color=["#3498db", "#e74c3c"], width=0.5)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{height} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Target Distribution: Heart Disease vs Normal", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    fig.savefig(reports_dir / "eda_target_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Numerical Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for idx, col in enumerate(NUMERIC_COLS):
        ax = axes[idx]
        sns.histplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            kde=True,
            palette={0: "#3498db", 1: "#e74c3c"},
            ax=ax,
            bins=20,
            alpha=0.6,
        )
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
    # Hide unused 6th subplot
    axes[5].axis("off")
    plt.tight_layout()
    fig.savefig(reports_dir / "eda_numeric_distributions.png", dpi=300)
    plt.close(fig)

    # 3. Categorical Variables by Target
    key_cats = ["ChestPainType", "ST_Slope", "ExerciseAngina", "Sex"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(key_cats):
        ax = axes[idx]
        sns.countplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            palette={0: "#3498db", 1: "#e74c3c"},
            ax=ax,
        )
        ax.set_title(f"{col} by Heart Disease Status", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(["Normal (0)", "Heart Disease (1)"], title="Diagnosis")
    plt.tight_layout()
    fig.savefig(reports_dir / "eda_categorical_by_target.png", dpi=300)
    plt.close(fig)

    # 4. Correlation Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = NUMERIC_COLS + [TARGET_COL]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap (Numerical Features & Target)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(reports_dir / "eda_correlation_matrix.png", dpi=300)
    plt.close(fig)


def plot_roc_curves(models_dict, X_test, y_test, save_path):
    """
    Plots test-set ROC curves for all models side-by-side with AUC scores.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, model), color in zip(models_dict.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2.2)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc = roc_auc_score(y_test, y_score)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test, save_path, model_name="Best Model"):
    """
    Plots normalized and raw confusion matrix for the winning model.
    Highlights false negatives for clinical safety consideration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array([
        [f"{cm[0,0]}\n({cm_norm[0,0]:.1%})\n[True Neg]", f"{cm[0,1]}\n({cm_norm[0,1]:.1%})\n[False Pos]"],
        [f"{cm[1,0]}\n({cm_norm[1,0]:.1%})\n[False Neg]", f"{cm[1,1]}\n({cm_norm[1,1]:.1%})\n[True Pos]"]
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Normal (0)", "Heart Disease (1)"],
        yticklabels=["Normal (0)", "Heart Disease (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted Diagnosis", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Diagnosis", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix: {model_name}\n(Prioritizing Low False Negatives)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(best_model, X_train, y_train, save_path):
    """
    Calculates and plots feature importance for tree models or coefficient magnitude
    for linear models, along with mutual information.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract feature names from fitted pipeline
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    # Transform training data to inspect engineered columns
    cleaned_X = best_model.named_steps["cleaner"].transform(X_train)
    transformed_X = preprocessor.transform(cleaned_X)

    # Retrieve feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out([c for c in CATEGORICAL_COLS if c != "FastingBS"]))
    feature_names = NUMERIC_COLS + encoded_cat_names + ["FastingBS", "Cholesterol_missing"]

    # Determine importances
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        metric_name = "Gini Feature Importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        metric_name = "Absolute Coefficient Magnitude"
    else:
        # Fallback to Mutual Information
        importances = mutual_info_classif(transformed_X, y_train, random_state=42)
        metric_name = "Mutual Information Score"

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=True).tail(12)  # Top 12 features

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color="#2b5c8f", edgecolor="black", height=0.6)
    ax.set_xlabel(metric_name, fontsize=11, fontweight="bold")
    ax.set_title(f"Top Clinical Predictive Factors ({metric_name})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
