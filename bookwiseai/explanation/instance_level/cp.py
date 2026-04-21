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
    logger.info("Running instance-level explanation of Ceteris-paribus Profiles method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']

    instance = test[features].iloc[0:1]
    logger.info(f"Instance to explain:\n{instance}\n")

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating Ceteris-paribus Profiles explanation for instance...")
    cp_instance = exp.predict_profile(instance)
    logger.info(f"\n{cp_instance.result}")

    con_features = ['lead_time', 'adr_clipped']
    cp_instance.plot(variables=con_features)

    fig_con = cp_instance.plot(variables=con_features, show=False)
    OUT_path_con = plot_path / "cp_con.png"
    fig_con.write_image(OUT_path_con)
    logger.info(f"Saved Ceteris-paribus Profiles continuous plot to: {OUT_path_con}")

    cat_features = ['hotel', 'market_segment']
    cp_instance.plot(variables=cat_features)

    fig_cat = cp_instance.plot(variables=cat_features, show=False)
    OUT_path_cat = plot_path / "cp_cat.png"
    fig_cat.write_image(OUT_path_cat)
    logger.info(f"Saved Ceteris-paribus Profiles categorical plot to: {OUT_path_cat}")

    logger.success("Instance-level explanation of Ceteris-paribus Profiles method complete.")


if __name__ == "__main__":
    app()
