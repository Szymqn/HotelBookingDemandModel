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
    logger.info("Running instance-level explanation of break down method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']

    instance = test[features].iloc[0:1]
    logger.info(f"Instance to explain:\n{instance}\n")

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating break down explanation for instance...")
    bd_instance = exp.predict_parts(
        instance,
        type="break_down"
    )
    logger.info(f"\n{bd_instance.result}")
    bd_instance.plot(show=True)

    fig = bd_instance.plot(show=False)
    OUT_path = plot_path / "break_down.png"
    fig.write_image(OUT_path)
    logger.info(f"Saved break down plot to: {OUT_path}")

    logger.info("Generating Ibreak down explanation for instance...")
    bd_interactions_instance = exp.predict_parts(
        instance,
        type="break_down_interactions",
        interaction_preference=10,
    )
    logger.info(f"\n{bd_interactions_instance.result}")
    bd_interactions_instance.plot(show=True)

    fig_interactions = bd_interactions_instance.plot(show=False)
    OUT_interactions_path = plot_path / "break_down_interactions.png"
    fig_interactions.write_image(OUT_interactions_path)
    logger.info(f"Saved break down plot to: {OUT_interactions_path}")

    logger.success("Instance-level explanation of break down method complete.")


if __name__ == "__main__":
    app()
