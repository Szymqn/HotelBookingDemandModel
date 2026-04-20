import typer
import joblib
import pandas as pd

from pathlib import Path
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from bookwiseai.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()

def run_lasso(X: pd.DataFrame, y: pd.Series) -> set:
    logger.info("Running LASSO...")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    selector = SelectFromModel(
        LogisticRegression()
    )
    selector.fit(X_scaled, y)
    selected = set(X_scaled.columns[selector.get_support()].tolist())
    logger.info(f"LASSO     : {len(selected)} features selected")
    return selected

def run_rfe_lr(X: pd.DataFrame, y: pd.Series) -> set:
    logger.info("Running RFE with Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    selector = RFE(
        estimator=LogisticRegression(),
        step=1,
    )
    selector.fit(X_scaled, y)
    selected = set(X_scaled.columns[selector.get_support()].tolist())
    logger.info(f"RFE-LR    : {len(selected)} features selected")
    return selected

def run_rfe_dt(X: pd.DataFrame, y: pd.Series) -> set:
    logger.info("Running RFE with Decision Tree...")
    selector = RFE(
        estimator=DecisionTreeClassifier(),
        step=1,
    )
    selector.fit(X, y)
    selected = set(X.columns[selector.get_support()].tolist())
    logger.info(f"RFE-RF    : {len(selected)} features selected")
    return selected

def run_rfe_rf(X: pd.DataFrame, y: pd.Series) -> set:
    logger.info("Running RFE with Random Forest...")
    selector = RFE(
        estimator=RandomForestClassifier(),
        step=1,
    )
    selector.fit(X, y)
    selected = set(X.columns[selector.get_support()].tolist())
    logger.info(f"RFE-RF    : {len(selected)} features selected")
    return selected

def run_rfe_xgb(X: pd.DataFrame, y: pd.Series) -> set:
    logger.info("Running RFE with XGBoost...")
    selector = RFE(
        estimator=XGBClassifier(),
        step=1,
    )
    selector.fit(X, y)
    selected = set(X.columns[selector.get_support()].tolist())
    logger.info(f"RFE-XGB   : {len(selected)} features selected")
    return selected

def load_r_selected(path: Path, method: str, X: pd.DataFrame) -> set:
    logger.info(f"Loading {method} results from {path}...")

    if not path.exists():
        logger.warning(f"{method} file not found at {path} — skipping")
        return set()

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        logger.warning(f"{method} file must have a 'feature' column — skipping")
        return set()

    all_features = set(X.columns)
    selected = set(df["feature"].tolist())

    skipped = selected - all_features
    if skipped:
        logger.warning(f"{method} features not in dataset (skipped): {skipped}")

    selected = selected & all_features
    logger.info(f"{method:<10}: {len(selected)} features loaded")
    return selected

@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    val_path: Path = PROCESSED_DATA_DIR / "valid.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    boruta_path: Path = EXTERNAL_DATA_DIR / "boruta_selected.csv",
    mdfs_path:   Path = EXTERNAL_DATA_DIR / "mdfs_selected.csv",
    min_votes: int = 3,
):
    logger.info("Performing feature selection...")

    train = pd.read_csv(train_path)
    TARGET = "is_canceled"
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    results = {
        "lasso": run_lasso(X_train, y_train),
        "rfe_lr": run_rfe_lr(X_train, y_train),
        "rfe_dt": run_rfe_dt(X_train, y_train),
        "rfe_rf": run_rfe_rf(X_train, y_train),
        "rfe_xgb": run_rfe_xgb(X_train, y_train),
        "boruta": load_r_selected(boruta_path, "boruta", X_train),
        "mdfs":   load_r_selected(mdfs_path, "mdfs", X_train),
    }

    for method, selected in results.items():
        path = PROCESSED_DATA_DIR / f"features_{method}.pkl"
        joblib.dump(sorted(selected), path)
        logger.info(f"Saved {method:<10} → {path.name} ({len(selected)} features)")

    votes = {col: 0 for col in X_train.columns}
    for method, selected in results.items():
        for col in selected:
            if col in votes:
                votes[col] += 1

    logger.info("\nFeature vote summary:")
    for feat, vote in sorted(votes.items(), key=lambda x: -x[1]):
        bar = "\u2588" * vote + "\u2591" * (len(results) - vote)
        logger.info(f"  {feat:<45} [{bar}] {vote}/{len(results)}")

    final_features = [f for f, v in votes.items() if v >= min_votes]
    logger.info(f"\nFinal: {len(final_features)}/{len(X_train.columns)} features kept (≥{min_votes} votes)")

    for path, name in [(train_path, "train_vote_fs"), (val_path, "valid_vote_fs"), (test_path, "test_vote_fs")]:
        df = pd.read_csv(path)
        df = df[final_features + [TARGET]]
        df.to_csv(path, index=False)
        logger.info(f"{name} saved: {df.shape}")

    joblib.dump(final_features, PROCESSED_DATA_DIR / "selected_features.pkl")
    logger.info("Selected features saved to processed/selected_features.pkl")

    logger.success("Feature selection complete.")


if __name__ == "__main__":
    app()
