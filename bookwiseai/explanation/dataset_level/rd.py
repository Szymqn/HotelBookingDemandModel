import joblib
import typer
import pandas as pd
import dalex as dx

from pathlib import Path
from loguru import logger

from bookwiseai.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

app = typer.Typer()

@app.command()
def main(
        test_path:     Path = PROCESSED_DATA_DIR / "test.csv",
        features_path: Path = PROCESSED_DATA_DIR / "features_voted.pkl",
        model_path:    Path = MODELS_DIR / "trained_models" / "voted_best_model.pkl",
        plot_path:     Path = FIGURES_DIR,
):
    logger.info("Running dataset-level explanation of Residual-diagnostics method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating Residual-diagnostics explanation for dataset...")
    rd = exp.model_diagnostics()
    logger.info(f"\n{rd.result}")
    rd.plot(show=True)

    fig = rd.plot(show=False)
    OUT_path = plot_path / "rd.png"
    fig.write_image(OUT_path)
    logger.info(f"Saved Residual-diagnostics plot to: {OUT_path}")

    rd.plot(variable='lead_time', yvariable='is_canceled', show=True)

    logger.success("Dataset-level explanation of Residual-diagnostics method complete.")


if __name__ == "__main__":
    app()
