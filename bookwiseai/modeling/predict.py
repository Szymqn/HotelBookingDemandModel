import typer
import joblib
import pandas as pd

from pathlib import Path
from loguru import logger

from bookwiseai.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
        processed_path: Path = PROCESSED_DATA_DIR,
        valid_path: Path = PROCESSED_DATA_DIR / "valid.csv",
        models_path: Path = MODELS_DIR / "trained_models",
        predictions_path: Path = MODELS_DIR / "predictions",
        models_path_list: list[Path] | None = None,
        features_path_list: list[Path] | None = None,
):
    if models_path_list is None:
        models_path_list = [
            Path("boruta_best_model.pkl"),
            Path("lasso_best_model.pkl"),
            Path("mdfs_best_model.pkl"),
            Path("rfedt_best_model.pkl"),
            Path("rfelr_best_model.pkl"),
            Path("rferf_best_model.pkl"),
            Path("rfexgb_best_model.pkl"),
            Path("voted_best_model.pkl"),
        ]
    assert models_path_list is not None and len(models_path_list) > 0,\
        "Please provide at least one model file in models_path_list"

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

    valid = pd.read_csv(valid_path).drop(columns=["is_canceled"])

    logger.info("Performing inference for model...")

    for features_path, model_path in zip(features_path_list, models_path_list):
        feature_set_name = Path(model_path).stem.split("_")[0]

        final_model_path = models_path / model_path
        model = joblib.load(final_model_path)

        final_features_path = processed_path / features_path
        selected_features = joblib.load(final_features_path)

        predictions = model.predict(valid[selected_features])

        predictions_df = pd.DataFrame({
            "pred": predictions,
        })
        predictions_file_path = predictions_path / f"{feature_set_name}_predictions.csv"
        predictions_df.to_csv(predictions_file_path, index=False)

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
