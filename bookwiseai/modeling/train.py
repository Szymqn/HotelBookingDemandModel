import typer
import joblib
import pandas as pd

from pathlib import Path
from loguru import logger

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from bookwiseai.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def get_models(scale_pos_weight: float) -> dict:
    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=1,
                        C=1.0,
                        solver="saga",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            auto_class_weights="Balanced",
            iterations=100,
            random_seed=42,
            verbose=False,
            train_dir=str(MODELS_DIR/"catboost_info"),
            allow_writing_files=True,
        ),
    }

    return models

@app.command()
def main(
    train_path:     Path = PROCESSED_DATA_DIR / "train.csv",
    val_path:       Path = PROCESSED_DATA_DIR / "valid.csv",
    processed_path: Path = PROCESSED_DATA_DIR,
    model_path:     Path = MODELS_DIR / "trained_models",
    model_summaries_path: Path = MODELS_DIR / "summaries",
    features_path_list: list[Path] | None = None,
    n_splits:      int  = 5,
) -> None:
    if features_path_list is None:
        features_path_list = [
            Path("features_boruta.pkl"),
            Path("features_lasso.pkl"),
            Path("features_mdfs.pkl"),
            Path("features_rfedt.pkl"),
            Path("features_rfelr.pkl"),
            Path("features_rferf.pkl"),
            Path("features_rfexgb.pkl"),
            Path("features_voted.pkl"),
        ]
    assert features_path_list is not None and len(features_path_list) > 0,\
        "Please provide at least one features file in features_path_list"

    logger.info("Loading data...")
    TARGET = "is_canceled"

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    for features_path in features_path_list:
        final_features_path = processed_path / features_path
        feature_set_name = features_path.stem.removeprefix("features_")

        selected_features = joblib.load(final_features_path)
        if not selected_features:
            logger.warning(f"No features loaded from {features_path.name} — skipping")
            continue
        else:
            logger.info(f"Using {len(selected_features)} features from {features_path.name}")

        X_train = train[selected_features]
        y_train = train[TARGET]
        X_val = val[selected_features]
        y_val = val[TARGET]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        models = get_models(scale_pos_weight)

        logger.info(f"Running {n_splits}-fold StratifiedKFold CV on train set...")
        cv_records = []

        for name, model in models.items():
            logger.info(f"  CV → {name}...")
            scores = cross_validate(
                model, X_train, y_train,
                cv=cv,
                scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
                n_jobs=-1,
            )
            record = {"model": name}
            for metric, values in scores.items():
                if metric.startswith("test_"):
                    metric_name = metric.replace("test_", "")
                    record[f"cv_{metric_name}_mean"] = values.mean()
                    record[f"cv_{metric_name}_std"] = values.std()

            cv_records.append(record)
            logger.info(
                f"    AUC: {record['cv_roc_auc_mean']:.4f} ± {record['cv_roc_auc_std']:.4f} | "
                f"F1: {record['cv_f1_mean']:.4f} ± {record['cv_f1_std']:.4f}"
            )

        logger.info("Evaluating on validation set...")
        val_records = []
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            val_record = {
                "model": name,
                "val_auc": roc_auc_score(y_val, y_pred_proba),
                "val_f1": f1_score(y_val, y_pred),
                "val_precision": precision_score(y_val, y_pred),
                "val_recall": recall_score(y_val, y_pred),
                "val_accuracy": accuracy_score(y_val, y_pred),
            }
            val_records.append(val_record)
            trained_models[name] = model

            logger.info(
                f"  {name:<25} "
                f"AUC: {val_record['val_auc']:.4f} | "
                f"F1: {val_record['val_f1']:.4f} | "
                f"Precision: {val_record['val_precision']:.4f} | "
                f"Recall: {val_record['val_recall']:.4f}"
            )

        cv_df = pd.DataFrame(cv_records)
        val_df = pd.DataFrame(val_records)
        results_df = cv_df.merge(val_df, on="model")

        results_out = model_summaries_path / f"{feature_set_name}_cv_results.csv"
        results_df.to_csv(results_out, index=False)
        logger.info(f"CV + Val results saved to {results_out}")

        best_name = val_df.loc[val_df["val_auc"].idxmax(), "model"]
        best_model = trained_models[best_name]
        best_model_out = model_path / f"{feature_set_name}_best_model.pkl"
        joblib.dump(best_model, best_model_out)
        logger.info(f"Best model: {best_name} (AUC: {val_df['val_auc'].max():.4f})")
        logger.info(f"Best model saved to {best_model_out}")

        for name, model in trained_models.items():
            model_filename = f"{feature_set_name}_{name}.pkl"
            joblib.dump(model, model_path / model_filename)
            logger.info(f"Saved {feature_set_name}_{name} → models/{model_filename}")

    logger.success("Model training complete.")


if __name__ == "__main__":
    app()
