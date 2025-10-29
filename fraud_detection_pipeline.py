"""
Fraud detection workflow script.

This script covers exploratory data analysis, feature selection, sampling strategies,
model training, hyperparameter tuning, ensemble learning, and detailed evaluation
for credit card fraud detection on highly imbalanced data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Handle optional XGBoost import gracefully.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_FEATURES = 15
N_JOBS = 1
CV_FOLDS = 3
MAX_MODELS_TO_TUNE = 3
DEFAULT_RANDOM_SEARCH_ITER = 12

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard - menor balanceado.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"


def ensure_output_dirs() -> None:
    """Ensure output directories exist before writing artifacts."""
    for directory in (OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column as target variable.")

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """Generate exploratory data analysis artifacts."""
    ensure_output_dirs()

    # Save descriptive statistics.
    descriptive_stats = df.describe().transpose()
    descriptive_stats.to_csv(REPORTS_DIR / "descriptive_statistics.csv", index=True)

    # Class distribution summary.
    class_counts = df["Class"].value_counts().sort_index()
    class_percent = class_counts / class_counts.sum() * 100
    class_summary = pd.DataFrame(
        {"count": class_counts, "percentage": class_percent.round(2)}
    )
    class_summary.to_csv(REPORTS_DIR / "class_distribution.csv", index=True)

    class_df = class_summary.reset_index().rename(columns={"index": "Class"})
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=class_df,
        x="Class",
        y="count",
        hue="Class",
        palette="viridis",
        legend=False,
    )
    plt.title("Distribuição das Classes")
    plt.xlabel("Classe (0 = Legítima, 1 = Fraude)")
    plt.ylabel("Número de Transações")
    for i, count in enumerate(class_summary["count"]):
        plt.text(i, count + class_counts.max() * 0.02, str(int(count)), ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close()

    # Class percentage pie chart.
    plt.figure(figsize=(6, 6))
    plt.pie(
        class_summary["count"],
        labels=["Legítima", "Fraude"],
        autopct="%1.2f%%",
        startangle=90,
        colors=["#4C72B0", "#DD8452"],
    )
    plt.title("Percentual de cada Classe")
    plt.savefig(PLOTS_DIR / "class_percentage.png")
    plt.close()

    # Amount distribution by class.
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="Amount", hue="Class", fill=True, common_norm=False)
    plt.title("Distribuição de Amount por Classe")
    plt.savefig(PLOTS_DIR / "amount_distribution_by_class.png")
    plt.close()

    # Time distribution by class.
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="Time", hue="Class", fill=True, common_norm=False)
    plt.title("Distribuição de Time por Classe")
    plt.savefig(PLOTS_DIR / "time_distribution_by_class.png")
    plt.close()

    # Correlation heatmap (features only).
    corr = df.drop(columns=["Class"]).corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=False,
        cbar_kws={"shrink": 0.6},
    )
    plt.title("Matriz de Correlação das Features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_correlation_heatmap.png")
    plt.close()

    # PCA visualization (first two components).
    features = df.drop(columns=["Class"])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(
        components, columns=["PC1", "PC2"], index=df.index
    ).join(df["Class"])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Class",
        palette={0: "#4C72B0", 1: "#DD8452"},
        alpha=0.7,
    )
    plt.title("PCA - Primeiras Duas Componentes")
    plt.savefig(PLOTS_DIR / "pca_scatter.png")
    plt.close()


def get_feature_sets(X: pd.DataFrame, y: pd.Series) -> Dict[str, List[str]]:
    """Compute different feature subsets for comparison."""
    feature_sets: Dict[str, List[str]] = {"all_features": X.columns.tolist()}

    k = min(TOP_K_FEATURES, X.shape[1])

    # SelectKBest.
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    kbest_features = X.columns[selector.get_support()].tolist()
    feature_sets["select_k_best"] = kbest_features

    # Recursive Feature Elimination with Logistic Regression.
    rfe_estimator = LogisticRegression(
        solver="liblinear", random_state=RANDOM_STATE, max_iter=1000
    )
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=k)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.get_support()].tolist()
    feature_sets["rfe_log_reg"] = rfe_features

    # Feature importance via Random Forest.
    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    rf_top_features = importances.sort_values(ascending=False).head(k).index.tolist()
    feature_sets["random_forest_importance"] = rf_top_features

    # Persist feature sets for transparency.
    (REPORTS_DIR / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2), encoding="utf-8"
    )

    return feature_sets


def get_samplers() -> Dict[str, Optional[object]]:
    """Return configured sampling strategies."""
    samplers: Dict[str, Optional[object]] = {
        "none": None,
        "smote": SMOTE(random_state=RANDOM_STATE),
        "random_undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
        "smoteenn": SMOTEENN(random_state=RANDOM_STATE),
    }
    return samplers


def get_models() -> Dict[str, ClassifierMixin]:
    """Instantiate base models for evaluation."""
    models: Dict[str, ClassifierMixin] = {
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "svm_rbf": SVC(probability=True, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=400,
        )

    return models


def build_pipeline(
    classifier: ClassifierMixin, sampler: Optional[object] = None
) -> ImbPipeline:
    """Create an imblearn pipeline for reproducible preprocessing."""
    steps: List[Tuple[str, object]] = []

    # Scaling before sampling stabilizes distance-based synthesis.
    steps.append(("scaler", StandardScaler()))

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


@dataclass
class EvaluationResult:
    """Container for model evaluation metrics."""

    feature_set: str
    sampler: str
    model_name: str
    tuned: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    f1_macro: float
    roc_auc: float
    confusion_matrix_path: str
    classification_report_path: str
    roc_curve_path: str
    pr_curve_path: str


def _get_probabilities(
    model: ClassifierMixin, X_test: pd.DataFrame
) -> np.ndarray:
    """Safely extract positive class probabilities."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_test)
        proba = (decision_scores - decision_scores.min()) / (
            decision_scores.max() - decision_scores.min() + 1e-8
        )
    else:
        # Fall back to predictions if probabilities are unavailable.
        proba = model.predict(X_test)
    return proba


def save_classification_artifacts(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    run_label: str,
) -> Tuple[str, str, str, str]:
    """Persist confusion matrix, classification report, ROC, and PR curves."""
    ensure_output_dirs()

    safe_label = run_label.replace(" ", "_").lower()

    cm = confusion_matrix(y_test, y_pred)
    cm_fig = plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legítima", "Fraude"],
        yticklabels=["Legítima", "Fraude"],
    )
    plt.title(f"Matriz de Confusão - {run_label}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    cm_path = PLOTS_DIR / f"confusion_matrix_{safe_label}.png"
    plt.tight_layout()
    cm_fig.savefig(cm_path)
    plt.close(cm_fig)

    # Classification report.
    report_text = classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"])
    report_path = REPORTS_DIR / f"classification_report_{safe_label}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {run_label}")
    plt.legend(loc="lower right")
    roc_path = PLOTS_DIR / f"roc_curve_{safe_label}.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Precision-Recall curve.
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_fig = plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall - {run_label}")
    plt.legend(loc="lower left")
    pr_path = PLOTS_DIR / f"precision_recall_curve_{safe_label}.png"
    plt.tight_layout()
    pr_fig.savefig(pr_path)
    plt.close(pr_fig)

    return (
        str(cm_path.relative_to(BASE_DIR)),
        str(report_path.relative_to(BASE_DIR)),
        str(roc_path.relative_to(BASE_DIR)),
        str(pr_path.relative_to(BASE_DIR)),
    )


def evaluate_model_pipeline(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set: str,
    sampler_name: str,
    model_name: str,
    tuned: bool,
) -> EvaluationResult:
    """Train a pipeline and return evaluation metrics with saved artifacts."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = _get_probabilities(pipeline, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    run_label = f"{feature_set}_{sampler_name}_{model_name}{'_tuned' if tuned else ''}"
    cm_path, report_path, roc_path, pr_path = save_classification_artifacts(
        y_test, y_pred, y_proba, run_label
    )

    return EvaluationResult(
        feature_set=feature_set,
        sampler=sampler_name,
        model_name=model_name,
        tuned=tuned,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f1_macro=f1_macro,
        roc_auc=roc_auc,
        confusion_matrix_path=cm_path,
        classification_report_path=report_path,
        roc_curve_path=roc_path,
        pr_curve_path=pr_path,
    )


def evaluate_all_models(
    feature_sets: Dict[str, List[str]],
    samplers: Dict[str, Optional[object]],
    models: Dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> List[EvaluationResult]:
    """Evaluate every combination of feature set, sampler, and model."""
    results: List[EvaluationResult] = []

    for feature_label, features in feature_sets.items():
        X_train_fs = X_train[features]
        X_test_fs = X_test[features]

        for sampler_label, sampler in samplers.items():
            for model_label, model in models.items():
                pipeline = build_pipeline(clone(model), sampler=sampler)
                result = evaluate_model_pipeline(
                    pipeline=pipeline,
                    X_train=X_train_fs,
                    y_train=y_train,
                    X_test=X_test_fs,
                    y_test=y_test,
                    feature_set=feature_label,
                    sampler_name=sampler_label,
                    model_name=model_label,
                    tuned=False,
                )
                results.append(result)

    return results


def export_results(results: Iterable[EvaluationResult], filename: str) -> pd.DataFrame:
    """Export evaluation results to CSV."""
    df = pd.DataFrame([asdict(result) for result in results])
    df.sort_values(by=["tuned", "recall", "f1_macro"], ascending=[True, False, False], inplace=True)
    df.to_csv(REPORTS_DIR / filename, index=False)
    return df


def get_param_grids() -> Dict[str, Dict[str, Iterable]]:
    """Return hyperparameter grids for each model."""
    param_grids: Dict[str, Dict[str, Iterable]] = {
        "random_forest": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        "svm_rbf": {
            "classifier__C": [0.5, 1, 5],
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf"],
        },
        "gradient_boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5],
        },
        "decision_tree": {
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["minkowski"],
        },
    }

    if XGBOOST_AVAILABLE:
        param_grids["xgboost"] = {
            "classifier__n_estimators": [300, 500],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
        }

    return param_grids


def identify_best_configuration_by_model(
    results_df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    """Select the configuration with the highest recall for each model."""
    best_configs: Dict[str, Dict[str, str]] = {}
    for model_name, group in results_df.groupby("model_name"):
        best_row = group.sort_values(by=["recall", "f1_macro"], ascending=[False, False]).iloc[0]
        best_configs[model_name] = {
            "feature_set": best_row["feature_set"],
            "sampler": best_row["sampler"],
        }
    return best_configs


def select_models_to_tune(
    results_df: pd.DataFrame, max_models: int
) -> List[str]:
    """Choose top-performing models (by recall then macro-F1) for tuning."""
    base_results = results_df[results_df["tuned"] == False]
    if base_results.empty:
        return []

    best_per_model = (
        base_results.sort_values(by=["recall", "f1_macro"], ascending=[False, False])
        .drop_duplicates(subset=["model_name"])
        .sort_values(by=["recall", "f1_macro"], ascending=[False, False])
    )

    return best_per_model["model_name"].head(max_models).tolist()


def tune_models(
    models: Dict[str, ClassifierMixin],
    param_grids: Dict[str, Dict[str, Iterable]],
    feature_sets: Dict[str, List[str]],
    samplers: Dict[str, Optional[object]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_configs: Dict[str, Dict[str, str]],
    models_to_tune: List[str],
) -> Tuple[List[EvaluationResult], Dict[str, Dict[str, object]]]:
    """Perform hyperparameter tuning for each model."""
    tuned_results: List[EvaluationResult] = []
    best_estimators: Dict[str, Dict[str, object]] = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in models.items():
        if model_name not in param_grids or model_name not in models_to_tune:
            continue

        config = best_configs[model_name]
        feature_label = config["feature_set"]
        sampler_label = config["sampler"]

        X_train_fs = X_train[feature_sets[feature_label]]
        X_test_fs = X_test[feature_sets[feature_label]]
        sampler = samplers[sampler_label]

        pipeline = build_pipeline(clone(model), sampler=sampler)
        param_grid = param_grids[model_name]
        full_grid_size = len(ParameterGrid(param_grid))
        n_iter = min(DEFAULT_RANDOM_SEARCH_ITER, full_grid_size)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=N_JOBS,
            verbose=1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train_fs, y_train)
        best_pipeline = search.best_estimator_

        tuned_result = evaluate_model_pipeline(
            pipeline=best_pipeline,
            X_train=X_train_fs,
            y_train=y_train,
            X_test=X_test_fs,
            y_test=y_test,
            feature_set=feature_label,
            sampler_name=sampler_label,
            model_name=model_name,
            tuned=True,
        )
        tuned_results.append(tuned_result)

        best_estimators[model_name] = {
            "pipeline": best_pipeline,
            "best_params": search.best_params_,
            "feature_set": feature_label,
            "sampler": sampler_label,
            "best_score": search.best_score_,
        }

    return tuned_results, best_estimators


def build_ensembles(
    tuned_estimators: Dict[str, Dict[str, object]],
    feature_sets: Dict[str, List[str]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> List[EvaluationResult]:
    """Construct and evaluate ensemble strategies using tuned classifiers."""
    if not tuned_estimators:
        return []

    # Choose the feature set used most frequently among tuned models.
    feature_usage = pd.Series(
        [meta["feature_set"] for meta in tuned_estimators.values()]
    ).value_counts()
    ensemble_feature_set = feature_usage.index[0]
    features = feature_sets[ensemble_feature_set]

    # Use SMOTE by default for ensembles to prioritize recall.
    sampler = SMOTE(random_state=RANDOM_STATE)

    X_train_fs = X_train[features]
    X_test_fs = X_test[features]

    X_train_res, y_train_res = sampler.fit_resample(X_train_fs, y_train)
    scaler = StandardScaler().fit(X_train_res)
    X_train_scaled = scaler.transform(X_train_res)
    X_test_scaled = scaler.transform(X_test_fs)

    evaluated_models: List[EvaluationResult] = []

    # Extract tuned classifiers.
    estimators_for_ensemble: List[Tuple[str, ClassifierMixin]] = []
    for model_name, meta in tuned_estimators.items():
        pipeline: ImbPipeline = meta["pipeline"]  # type: ignore
        classifier = clone(pipeline.named_steps["classifier"])
        estimators_for_ensemble.append((model_name, classifier))

    # Voting ensemble.
    voting_clf = VotingClassifier(estimators=estimators_for_ensemble, voting="soft")
    voting_clf.fit(X_train_scaled, y_train_res)
    y_pred = voting_clf.predict(X_test_scaled)
    y_proba = _get_probabilities(voting_clf, X_test_scaled)
    run_label = f"{ensemble_feature_set}_smote_voting_classifier_tuned"
    cm_path, report_path, roc_path, pr_path = save_classification_artifacts(
        y_test, y_pred, y_proba, run_label
    )
    voting_result = EvaluationResult(
        feature_set=ensemble_feature_set,
        sampler="smote",
        model_name="voting_classifier",
        tuned=True,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        confusion_matrix_path=cm_path,
        classification_report_path=report_path,
        roc_curve_path=roc_path,
        pr_curve_path=pr_path,
    )
    evaluated_models.append(voting_result)

    # Stacking ensemble with logistic regression meta-classifier.
    stacking_clf = StackingClassifier(
        estimators=estimators_for_ensemble,
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        stack_method="predict_proba",
        n_jobs=N_JOBS,
        passthrough=False,
    )

    stacking_clf.fit(X_train_scaled, y_train_res)
    y_pred = stacking_clf.predict(X_test_scaled)
    y_proba = _get_probabilities(stacking_clf, X_test_scaled)
    run_label = f"{ensemble_feature_set}_smote_stacking_classifier_tuned"
    cm_path, report_path, roc_path, pr_path = save_classification_artifacts(
        y_test, y_pred, y_proba, run_label
    )
    stacking_result = EvaluationResult(
        feature_set=ensemble_feature_set,
        sampler="smote",
        model_name="stacking_classifier",
        tuned=True,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        confusion_matrix_path=cm_path,
        classification_report_path=report_path,
        roc_curve_path=roc_path,
        pr_curve_path=pr_path,
    )
    evaluated_models.append(stacking_result)

    return evaluated_models


def compare_sampling_strategies(results_df: pd.DataFrame) -> None:
    """Create visualization comparing recall across sampling strategies."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x="sampler",
        y="recall",
        hue="feature_set",
    )
    plt.title("Recall por Estratégia de Amostragem e Seleção de Features")
    plt.xlabel("Estratégia de Amostragem")
    plt.ylabel("Recall (Fraudes)")
    plt.legend(title="Conjunto de Features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recall_comparison_sampling_feature_selection.png")
    plt.close()


def compare_models(results_df: pd.DataFrame, tuned: bool, metric: str = "recall") -> None:
    """Generate model comparison bar chart."""
    filtered = results_df[results_df["tuned"] == tuned]

    if filtered.empty:
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=filtered,
        x="model_name",
        y=metric,
        hue="sampler",
    )
    title_state = "Tuned" if tuned else "Base"
    plt.title(f"{metric.capitalize()} por Modelo ({title_state})")
    plt.xlabel("Modelo")
    plt.ylabel(metric.capitalize())
    plt.legend(title="Amostragem")
    plt.tight_layout()
    filename = f"{metric}_comparison_{'tuned' if tuned else 'base'}.png"
    plt.savefig(PLOTS_DIR / filename)
    plt.close()


def main() -> None:
    """Run the complete fraud detection workflow."""
    ensure_output_dirs()

    df = load_dataset(DATA_PATH)
    perform_eda(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Ensure binary target.
    target_type = type_of_target(y)
    if target_type != "binary":
        raise ValueError(f"Expected binary target, found {target_type}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    feature_sets = get_feature_sets(X_train, y_train)
    samplers = get_samplers()
    models = get_models()

    base_results = evaluate_all_models(
        feature_sets=feature_sets,
        samplers=samplers,
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    base_df = export_results(base_results, "model_performance_base.csv")

    # Visual comparison for base models.
    compare_sampling_strategies(base_df[base_df["tuned"] == False])
    compare_models(base_df, tuned=False, metric="recall")
    compare_models(base_df, tuned=False, metric="f1_macro")

    best_configs = identify_best_configuration_by_model(base_df)
    param_grids = get_param_grids()
    models_to_tune = select_models_to_tune(base_df, MAX_MODELS_TO_TUNE)
    if models_to_tune:
        print(f"Modelos selecionados para tuning: {models_to_tune}")
    else:
        print("Nenhum modelo selecionado para tuning; pulando etapa de otimização.")
    tuned_results, tuned_estimators = tune_models(
        models=models,
        param_grids=param_grids,
        feature_sets=feature_sets,
        samplers=samplers,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        best_configs=best_configs,
        models_to_tune=models_to_tune,
    )

    tuned_df = export_results(tuned_results, "model_performance_tuned.csv")
    compare_models(tuned_df, tuned=True, metric="recall")
    compare_models(tuned_df, tuned=True, metric="f1_macro")

    ensemble_results = build_ensembles(
        tuned_estimators=tuned_estimators,
        feature_sets=feature_sets,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    if ensemble_results:
        ensemble_df = export_results(
            ensemble_results, "ensemble_performance.csv"
        )
        compare_models(ensemble_df, tuned=True, metric="recall")

    print("Pipeline concluído. Consulte a pasta 'outputs' para resultados detalhados.")


if __name__ == "__main__":
    main()
