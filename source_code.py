import pandas as pd
import numpy as np
import os

# --- PASTE THE FOLDER PATH HERE, with all the data sets ---
BASE_PATH = "project-7-files/"

def load_and_merge_data(subset='learn'):
    """
    Loads and merges all datasets for a given subset ('learn' or 'test').
    """
    print(f"--- Processing {subset.upper()} dataset ---")
    
    # 1. Load Main Dataset
    # We load Insee as string to preserve leading zeros as per instructions
    main_file = os.path.join(BASE_PATH, f"{subset}_dataset.csv")
    df = pd.read_csv(main_file, dtype={'Insee': str, 'Person_id': str})
    insee = df["Insee"].astype(str).str.strip()
    insee_valid = insee.str.match(r"^(\d{5}|2A\d{3}|2B\d{3})$")
    insee_invalid = ~insee_valid
    sample_invalid = (
        insee[insee_invalid]
        .dropna()
        .drop_duplicates()
        .head(10)
        .tolist()
    )
    print(
        "Sanity check (INSEE): "
        f"{insee_invalid.sum()} invalid codes "
        f"(sample: {sample_invalid})."
    )
    
    # 2. Add Job-Related Data
    emp_contract_file = os.path.join(BASE_PATH, f"{subset}_dataset_EMP_CONTRACT.csv")
    if os.path.exists(emp_contract_file):
        df_emp = pd.read_csv(emp_contract_file, dtype={'Person_id': str})
        df = df.merge(df_emp, on='Person_id', how='left')
        
    job_desc_file = os.path.join(BASE_PATH, f"{subset}_dataset_job.csv")
    if os.path.exists(job_desc_file):
        df_job = pd.read_csv(job_desc_file, dtype={'Person_id': str, 'job_dep': str})
        df_job = df_job.rename(columns={c: f"job_{c}" for c in df_job.columns if c != 'Person_id'})
        df = df.merge(df_job, on='Person_id', how='left')

    # 3. Add Retirement-Related Data
    retired_former_file = os.path.join(BASE_PATH, f"{subset}_dataset_retired_former.csv")
    if os.path.exists(retired_former_file):
        df_ret_former = pd.read_csv(retired_former_file, dtype={'Person_id': str})
        df = df.merge(df_ret_former, on='Person_id', how='left')
        
    retired_jobs_file = os.path.join(BASE_PATH, f"{subset}_dataset_retired_jobs.csv")
    if os.path.exists(retired_jobs_file):
        df_ret_jobs = pd.read_csv(retired_jobs_file, dtype={'Person_id': str, 'job_dep': str, 'PREVIOUS_DEP': str})
        df_ret_jobs = df_ret_jobs.rename(columns={c: f"ret_job_{c}" for c in df_ret_jobs.columns if c != 'Person_id'})
        df = df.merge(df_ret_jobs, on='Person_id', how='left')
        
    retired_pension_file = os.path.join(BASE_PATH, f"{subset}_dataset_retired_pension.csv")
    if os.path.exists(retired_pension_file):
        df_ret_pension = pd.read_csv(retired_pension_file, dtype={'Person_id': str})
        df = df.merge(df_ret_pension, on='Person_id', how='left')

    # 4. Add Sports Data
    sport_file = os.path.join(BASE_PATH, f"{subset}_dataset_sport.csv")
    if os.path.exists(sport_file):
        df_sport = pd.read_csv(sport_file, dtype={'Person_id': str})
        df = df.merge(df_sport, on='Person_id', how='left')
        df['SPORTS'] = df['SPORTS'].fillna('NONE')

    # 5. Integrate Profession Hierarchy (PCS-ESE 2017)
    code_map_path = os.path.join(BASE_PATH, 'code_work_desc_map.csv')
    if os.path.exists(code_map_path):
        work_map = pd.read_csv(code_map_path, dtype={'N3': str})
        if 'job_work_desc' in df.columns:
            df = df.merge(work_map.rename(columns={'N3': 'job_work_desc', 'N1': 'job_N1', 'N2': 'job_N2'}), 
                          on='job_work_desc', how='left')
        if 'ret_job_work_desc' in df.columns:
            df = df.merge(work_map.rename(columns={'N3': 'ret_job_work_desc', 'N1': 'ret_job_N1', 'N2': 'ret_job_N2'}), 
                          on='ret_job_work_desc', how='left')

    # 6. Geographical Enrichment
    df['DEP'] = df['Insee'].str[:2]
    dept_path = os.path.join(BASE_PATH, 'departments.csv')
    reg_path = os.path.join(BASE_PATH, 'regions.csv')
    if os.path.exists(dept_path) and os.path.exists(reg_path):
        depts = pd.read_csv(dept_path, dtype={'DEP': str})
        regs = pd.read_csv(reg_path)
        geo_info = depts.merge(regs, on='Reg', how='left')
        df = df.merge(geo_info, on='DEP', how='left')

    # 7. Additional Logic & Sanity Checks
    job_cols = [c for c in df.columns if c.startswith('job_')]
    df[job_cols] = df[job_cols].astype("object")
    if "job_CONTRACT_TYPE" in df.columns and "EMP_CONTRACT" in df.columns:
        emp_to_job = {
            "EC-1-1": "APP",
            "EC-1-2": "TTP",
            "EC-1-3": "AUT",
            "EC-1-4": "AUT",
            "EC-1-5": "CDD",
            "EC-1-6": "CDI",
        }
        normalized_emp = df["EMP_CONTRACT"].map(emp_to_job)
        comparable_mask = df["job_CONTRACT_TYPE"].notna() & normalized_emp.notna()
        mismatches = (
            df.loc[comparable_mask, "job_CONTRACT_TYPE"].astype(str)
            != normalized_emp[comparable_mask].astype(str)
        ).sum()
        if comparable_mask.sum() > 0:
            print(
                "Sanity check (normalized): "
                f"{mismatches} / {comparable_mask.sum()} inconsistent "
                "between job_CONTRACT_TYPE and EMP_CONTRACT."
            )
        df.loc[comparable_mask, "job_CONTRACT_TYPE"] = normalized_emp[comparable_mask]
    df.loc[df['ACT'].isin(['ACT-2-1', 'ACT-2-2']), job_cols] = df.loc[df['ACT'].isin(['ACT-2-1', 'ACT-2-2']), job_cols].fillna('RETIRED_OR_UNEMP')

    print(f"Finished {subset} merge. Shape: {df.shape}")
    return df

# Perform Merging
df_learn = load_and_merge_data('learn')
df_test = load_and_merge_data('test')

# --- SAVE TO THE BASE_PATH TO AVOID PERMISSION ERRORS ---
df_learn.to_csv(os.path.join(BASE_PATH, 'cleaned_learn_dataset.csv'), index=False)
df_test.to_csv(os.path.join(BASE_PATH, 'cleaned_test_dataset.csv'), index=False)

print(f"\nCleaning complete. Files saved in: {BASE_PATH}")

"""Single-file ML pipeline (combines all .py files from the src folder)."""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False

# =============================
# Configuration
# =============================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PLOT_STYLE = "seaborn-v0_8-paper"

CLEANED_DATA_DIR = Path(BASE_PATH)
TRAIN_DATA_PATH = CLEANED_DATA_DIR / "cleaned_learn_dataset.csv"
TEST_DATA_PATH = CLEANED_DATA_DIR / "cleaned_test_dataset.csv"

PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
SMALL_MACHINE = os.environ.get("SMALL_MACHINE", "1") == "1"

DROP_COLUMNS = ["Person_id", "target", "Insee"]
TARGET_COLUMN = "target"
ID_COLUMN = "Person_id"

DTYPE_SPEC = {
    "Insee": str,
    "Person_id": str,
    "DEP": str,
    "Reg": str,
}

# =============================
# Data
# =============================

def load_training_data() -> pd.DataFrame:
    """Load the training dataset."""
    return pd.read_csv(TRAIN_DATA_PATH, dtype=DTYPE_SPEC, low_memory=False)


def load_test_data() -> pd.DataFrame:
    """Load the test dataset."""
    return pd.read_csv(TEST_DATA_PATH, dtype=DTYPE_SPEC, low_memory=False)


def get_target_distribution(df: pd.DataFrame) -> pd.Series:
    """Return the target class distribution."""
    return df[TARGET_COLUMN].value_counts(normalize=True)


def get_data_info(df: pd.DataFrame, name: Optional[str] = None) -> dict:
    """Return dataset information."""
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    if name:
        info["name"] = name
    return info

# =============================
# Feature engineering
# =============================


class FeatureEncoder:
    """Handle categorical encoding with train/test consistency."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder: Optional[LabelEncoder] = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit encoders on training data and transform features.

        Args:
            df: Training DataFrame with the target column

        Returns:
            Tuple (encoded X, encoded y)
        """
        X = df.drop(
            columns=[c for c in DROP_COLUMNS if c in df.columns],
            errors="ignore",
        ).copy()

        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df[TARGET_COLUMN])

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype(str)
            self.encoders[col] = LabelEncoder()
            X[col] = self.encoders[col].fit_transform(X[col])

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using the fitted encoders.

        Args:
            df: Test DataFrame

        Returns:
            Encoded X DataFrame
        """
        X = df.drop(
            columns=[c for c in DROP_COLUMNS if c in df.columns],
            errors="ignore",
        ).copy()

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype(str)
            if col in self.encoders:
                le = self.encoders[col]
                X[col] = X[col].map(
                    lambda s, le=le: s if s in le.classes_ else le.classes_[0]
                )
                X[col] = le.transform(X[col])

        return X

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert predicted labels back to original classes."""
        return self.target_encoder.inverse_transform(y)

    def get_feature_names(self) -> list:
        """Return the list of encoded feature names."""
        return list(self.encoders.keys())

# =============================
# Models
# =============================


def get_models() -> Dict[str, Any]:
    """Return a dictionary of models to compare."""
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        print("XGBoost not installed: model skipped during cross-validation.")
    return models


def cross_validate_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on all models with hyperparameter tuning.

    Args:
        X: Encoded features
        y: Encoded labels

    Returns:
        Dictionary {model_name: {metric: mean, best_params: dict, best_estimator: model}}
    """
    models = get_models()
    results = {}
    grid_folds = 3 if SMALL_MACHINE else CV_FOLDS
    cv = StratifiedKFold(n_splits=grid_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    param_grids_full = {
        "Decision Tree": {
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "Hist Gradient Boosting": {
            "max_depth": [None, 6, 10],
            "learning_rate": [0.05, 0.1],
            "max_iter": [200, 400],
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }
    param_grids_small = {
        "Decision Tree": {
            "max_depth": [None, 6],
            "min_samples_split": [2, 5],
        },
        "Random Forest": {
            "n_estimators": [150],
            "max_depth": [None, 12],
            "min_samples_split": [2],
        },
        "Hist Gradient Boosting": {
            "max_depth": [None, 6],
            "learning_rate": [0.1],
            "max_iter": [150],
        },
        "XGBoost": {
            "n_estimators": [200],
            "max_depth": [5],
            "learning_rate": [0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        },
    }
    if SMALL_MACHINE:
        param_grids = param_grids_small
        models = {
            name: model
            for name, model in models.items()
            if name in {"Decision Tree", "Hist Gradient Boosting", "XGBoost", "Random Forest"}
        }
    else:
        param_grids = param_grids_full

    for name, model in models.items():
        grid = param_grids.get(name, {})
        search = GridSearchCV(
            model,
            grid,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        scores = cross_validate(best_model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy": scores["test_accuracy"].mean(),
            "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
            "f1_macro": scores["test_f1_macro"].mean(),
            "f1_weighted": scores["test_f1_weighted"].mean(),
            "best_params": search.best_params_,
            "best_estimator": best_model,
        }
        print(
            f"{name}: "
            f"acc={results[name]['accuracy']:.4f} | "
            f"bal_acc={results[name]['balanced_accuracy']:.4f} | "
            f"f1_macro={results[name]['f1_macro']:.4f} | "
            f"f1_weighted={results[name]['f1_weighted']:.4f} | "
            f"best_params={results[name]['best_params']}"
        )

    return results


def set_plot_style() -> None:
    """Apply a clean, academic plotting style."""
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
            "savefig.dpi": 150,
        }
    )


def plot_cv_scores(cv_results: Dict[str, Dict[str, float]]) -> None:
    """Save a bar chart of cross-validation metrics by model."""
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    models = list(cv_results.keys())
    values = np.array([[cv_results[m][k] for k in metrics] for m in models])

    x = np.arange(len(models))
    width = 0.2
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[:, i], width, label=metric)
    plt.xticks(x + width * 1.5, models, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Cross-Validation Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cv_scores.png", dpi=150)
    plt.close()


def select_best_model(cv_results: Dict[str, Dict[str, float]]) -> str:
    """Return the best model name based on mean balanced accuracy."""
    return max(cv_results, key=lambda name: cv_results[name]["balanced_accuracy"])


def train_final_model(
    model: Any, X: np.ndarray, y: np.ndarray
) -> Tuple[Any, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the final model with a train/validation split.

    Args:
        model: Model to train
        X: Encoded features
        y: Encoded labels

    Returns:
        Tuple (trained model, validation accuracy, confusion matrix, X_val, y_val, val_preds)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    cm = confusion_matrix(y_val, val_preds)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return model, accuracy, cm, X_val, y_val, val_preds


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save confusion matrix plot."""
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    disp.figure_.tight_layout()
    disp.figure_.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(disp.figure_)


def plot_roc_curve(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Save ROC curve and report AUC for binary classification."""
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_val)
    else:
        return
    auc = roc_auc_score(y_val, scores)
    RocCurveDisplay.from_predictions(y_val, scores)
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()


def plot_precision_recall_by_class(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray
) -> None:
    """Save precision/recall bar chart by class."""
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names))
    )
    x = np.arange(len(class_names))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, precision, width, label="precision")
    plt.bar(x + width / 2, recall, width, label="recall")
    plt.xticks(x, class_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Precision/Recall by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_by_class.png", dpi=150)
    plt.close()


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Trained model
        X: Features
        y: True labels

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions),
        "classification_report": classification_report(y, predictions),
    }

# =============================
# Predictions
# =============================


def generate_predictions(
    model: Any,
    X_test: pd.DataFrame,
    df_test: pd.DataFrame,
    encoder: FeatureEncoder,
) -> pd.DataFrame:
    """
    Generate predictions for the test set.

    Args:
        model: Trained model
        X_test: Encoded test features
        df_test: Original test DataFrame (for Person_id)
        encoder: Encoder to convert labels back

    Returns:
        DataFrame with predictions
    """
    predictions_encoded = model.predict(X_test)
    predictions = encoder.inverse_transform_target(predictions_encoded)

    submission = pd.DataFrame(
        {
            ID_COLUMN: df_test[ID_COLUMN],
            TARGET_COLUMN: predictions,
        }
    )
    return submission


def save_predictions(submission: pd.DataFrame, output_path: Optional[Path] = None) -> str:
    """
    Save predictions to CSV.

    Args:
        submission: DataFrame with predictions
        output_path: Output file path (optional, default: PREDICTIONS_PATH)

    Returns:
        Saved file path
    """
    path = output_path if output_path else PREDICTIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return str(path)

# =============================
# Main
# =============================


def main() -> None:
    """Main entry point for the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="ML pipeline for socio-demographic category prediction"
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=f"Output file path (default: {PREDICTIONS_PATH})",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else PREDICTIONS_PATH

    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    df_train = load_training_data()
    df_test = load_test_data()

    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"\nClass distribution:\n{get_target_distribution(df_train)}")

    print("\n" + "=" * 50)
    print("FEATURE PREPARATION")
    print("=" * 50)
    encoder = FeatureEncoder()
    X_train, y_train = encoder.fit_transform(df_train)
    X_test = encoder.transform(df_test)

    print(f"Number of features: {X_train.shape[1]}")

    cv_results = None
    if not args.skip_cv:
        print("\n" + "=" * 50)
        print("MODEL COMPARISON (5-Fold CV)")
        print("=" * 50)
        cv_results = cross_validate_models(X_train, y_train)
        plot_cv_scores(cv_results)

    print("\n" + "=" * 50)
    print("FINAL MODEL TRAINING")
    print("=" * 50)
    if args.skip_cv:
        final_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        print("Cross-validation skipped: using Hist Gradient Boosting model.")
    else:
        best_model_name = select_best_model(cv_results)
        final_model = cv_results[best_model_name]["best_estimator"]
        print(f"\nBest model (CV): {best_model_name}")
        print(
            "Expected CV performance (balanced_accuracy): "
            f"{cv_results[best_model_name]['balanced_accuracy']:.4f}"
        )

    model, accuracy, _, X_val, y_val, val_preds = train_final_model(
        final_model, X_train, y_train
    )
    plot_confusion_matrix(y_val, val_preds)
    plot_precision_recall_by_class(
        y_val, val_preds, encoder.target_encoder.classes_
    )
    if len(encoder.target_encoder.classes_) == 2:
        plot_roc_curve(model, X_val, y_val)

    print("\n" + "=" * 50)
    print("PREDICTION GENERATION")
    print("=" * 50)
    submission = generate_predictions(model, X_test, df_test, encoder)
    saved_path = save_predictions(submission, output_path)

    print(f"Predictions saved: {saved_path}")
    print(f"Number of predictions: {len(submission)}")
    print(f"\nPrediction distribution:\n{submission['target'].value_counts()}")


if __name__ == "__main__":
    main()
