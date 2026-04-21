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
    logger.info("Running dataset-level explanation of local-dependence and accumulated-local profiles method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']
    con_features = ['lead_time', 'adr_clipped']

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating local-dependence explanation for dataset...")
    ld = exp.model_profile(type='conditional')
    ld.result['_label_'] = 'LD profiles'
    logger.info(f"\n{ld.result}")
    ld.plot(variables=con_features, show=True)

    fig_ld = ld.plot(variables=con_features, show=False)
    OUT_ld_path = plot_path / "ld.png"
    fig_ld.write_image(OUT_ld_path)
    logger.info(f"Saved local-dependence plot to: {OUT_ld_path}")

    logger.info("Generating accumulated-local profiles explanation for dataset...")
    al = exp.model_profile(type='accumulated')
    al.result['_label_'] = 'AL profiles'
    logger.info(f"\n{al.result}")
    al.plot(variables=con_features, show=True)

    fig_al = al.plot(variables=con_features, show=False)
    OUT_al_path = plot_path / "al.png"
    fig_al.write_image(OUT_al_path)
    logger.info(f"Saved local-dependence plot to: {OUT_al_path}")

    logger.info("Generating combined-local profiles explanation for dataset...")
    al.plot(objects=[ld], variables=con_features, show=True)

    fig_ld_al = al.plot(ld, variables=con_features, show=False)
    OUT_ld_al_path = plot_path / "ld_al_combined.png"
    fig_ld_al.write_image(OUT_ld_al_path)
    logger.info(f"Saved local-dependence plot to: {OUT_ld_al_path}")

    logger.success("Dataset-level explanation of local-dependence and accumulated-local profiles methods complete.")


if __name__ == "__main__":
    app()
