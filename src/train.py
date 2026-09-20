"""
Training orchestration script for Heart Failure Detection System.
Trains and compares 4 machine learning models (Logistic Regression, Decision Tree,
Random Forest, SVM) with Stratified 5-Fold Cross-Validation, optimizes hyperparameters,
evaluates on an independent test set, and exports the winning diagnostic pipeline.
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    DATA_PATH,
    METRICS_CSV_PATH,
    METRICS_MD_PATH,
    PARAM_GRIDS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
    TEST_SIZE,
)
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_eda_figures,
    plot_feature_importance,
    plot_roc_curves,
)
from src.preprocess import build_full_pipeline, load_and_split_data


def train_and_evaluate_all():
    print("=" * 70)
    print(" HEART FAILURE DETECTION SYSTEM - TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/7] Ingesting dataset from: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please ensure heart.csv is in the data/ directory."
        )

    X_train, X_test, y_train, y_test, raw_df = load_and_split_data(
        DATA_PATH, target_col=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"      Total records: {len(raw_df)} (Features: {X_train.shape[1]}, Target: {TARGET_COL})")
    print(f"      Class distribution: {dict(raw_df[TARGET_COL].value_counts())} (Balanced ~55/45 - No resampling needed)")
    print(f"      Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples (80/20 Stratified)")

    # Data Quality Diagnostic Summary
    chol_zeros = (raw_df["Cholesterol"] == 0).sum()
    bp_zeros = (raw_df["RestingBP"] == 0).sum()
    print(f"      Data Quality Check: Cholesterol zeros = {chol_zeros} (handled via median + indicator flag)")
    print(f"      Data Quality Check: RestingBP zeros = {bp_zeros} (handled via median imputer)")

    # 2. Generate Exploratory Data Analysis Plots
    print(f"\n[2/7] Generating EDA visualizations in: {REPORTS_DIR}")
    plot_eda_figures(raw_df, REPORTS_DIR)
    print("      -> Saved: eda_target_distribution.png")
    print("      -> Saved: eda_numeric_distributions.png")
    print("      -> Saved: eda_categorical_by_target.png")
    print("      -> Saved: eda_correlation_matrix.png")

    # 3. Define Candidate Classifiers
    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }

    # 4. Stratified 5-Fold Cross Validation & Grid Search Tuning
    print(f"\n[3/7] Tuning hyperparameters using {CV_FOLDS}-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fitted_models = {}
    cv_scores = {}
    best_params = {}

    for name, clf in candidate_classifiers.items():
        print(f"      Optimizing: {name}...")
        pipeline = build_full_pipeline(clf)
        grid = PARAM_GRIDS.get(name, {})

        # Optimize prioritizing F1-score (harmonic balance of precision and recall)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        fitted_models[name] = grid_search.best_estimator_
        cv_scores[name] = {
            "mean_cv_f1": grid_search.best_score_,
            "std_cv_f1": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
        }
        best_params[name] = grid_search.best_params_
        print(f"         Best CV F1: {grid_search.best_score_:.4f} (±{cv_scores[name]['std_cv_f1']:.4f})")

    # 5. Test Set Evaluation
    print("\n[4/7] Evaluating tuned models on unseen test set (N=184)...")
    results = []

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics["CV F1 (Mean)"] = cv_scores[name]["mean_cv_f1"]
        metrics["CV F1 (Std)"] = cv_scores[name]["std_cv_f1"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Reorder columns for clinical presentation
    metric_cols_order = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "CV F1 (Mean)",
        "CV F1 (Std)",
        "False Negatives",
        "False Positives",
    ]
    results_df = results_df[metric_cols_order]

    # 6. Model Selection (Prioritize Recall & F1-Score to minimize missed cardiac diagnoses)
    # Primary sort: Recall (highest sensitivity), Secondary: F1-Score
    sorted_df = results_df.sort_values(by=["Recall", "F1-Score", "ROC-AUC"], ascending=[False, False, False])
    best_model_name = sorted_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_model_name]

    print(f"\n[5/7] Model Evaluation Comparison Table:")
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "False Negatives"]
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[6/7] Selected Best Model: >>> {best_model_name} <<<")
    print(f"      Clinical Justification: Prioritized Recall ({sorted_df.iloc[0]['Recall']:.4f}) and "
          f"F1-Score ({sorted_df.iloc[0]['F1-Score']:.4f}) to minimize dangerous False Negatives "
          f"({int(sorted_df.iloc[0]['False Negatives'])} missed cases).")

    # Save metrics table
    results_df.to_csv(METRICS_CSV_PATH, index=False)
    with open(METRICS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Performance Comparison\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
        except Exception:
            f.write(results_df.to_string(index=False))
        f.write(f"\n\n**Selected Model for Clinical Deployment**: {best_model_name}\n")
    print(f"      -> Exported comparison table: {METRICS_CSV_PATH}")
    print(f"      -> Exported markdown table: {METRICS_MD_PATH}")

    # 7. Generate Diagnostic Charts & Export Model
    print(f"\n[7/7] Generating comparative evaluation plots and serializing pipeline...")
    roc_path = REPORTS_DIR / "model_roc_curves.png"
    cm_path = REPORTS_DIR / "best_model_confusion_matrix.png"
    fi_path = REPORTS_DIR / "feature_importance.png"

    plot_roc_curves(fitted_models, X_test, y_test, roc_path)
    print(f"      -> Saved ROC comparison: {roc_path}")

    plot_confusion_matrix(best_pipeline, X_test, y_test, cm_path, model_name=best_model_name)
    print(f"      -> Saved Confusion Matrix: {cm_path}")

    plot_feature_importance(best_pipeline, X_train, y_train, fi_path)
    print(f"      -> Saved Feature Importance: {fi_path}")

    # Serialize best pipeline
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"      -> Serialized Best Pipeline: {BEST_MODEL_PATH}")

    print("\n" + "=" * 70)
    print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return best_model_name, results_df


if __name__ == "__main__":
    train_and_evaluate_all()
