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
    logger.info("Running dataset-level explanation of Partial-dependence profiles method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating Partial-dependence profiles explanation for dataset...")
    logger.info("Partial-dependence profiles for continuous features...")
    pdp_con = exp.model_profile(variables=['lead_time', 'adr_clipped'])
    logger.info(f"\n{pdp_con.result}")
    pdp_con.plot(show=True)

    fig_con = pdp_con.plot(show=False)
    OUT_path_con = plot_path / "pdp_con.png"
    fig_con.write_image(OUT_path_con)
    logger.info(f"Saved Partial-dependence profiles continuous plot to: {OUT_path_con}")

    pdp_con.plot(geom='profiles')

    fig_con_top = pdp_con.plot(geom='profiles', show=False)
    OUT_path_con_top = plot_path / "pdp_con_top.png"
    fig_con_top.write_image(OUT_path_con_top)
    logger.info(f"Saved Partial-dependence top profiles continuous plot to: {OUT_path_con_top}")

    # Test dataset not contain categorical data
    # logger.info("Partial-dependence profiles for categorical features...")
    # pdp_cat = exp.model_profile(variable_type='categorical')
    # logger.info(f"\n{pdp_cat.result}")
    # cat_features = ['hotel', 'market_segment']
    # pdp_cat.plot(cat_features, show=True)

    # fig_cat = pdp_cat.plot(cat_features, show=False)
    # OUT_path_cat = plot_path / "pdp_cat.png"
    # fig_cat.write_image(OUT_path_cat)
    # logger.info(f"Saved Partial-dependence profiles categorical plot to: {OUT_path_cat}")

    logger.success("Dataset-level explanation of Partial-dependence profiles method complete.")


if __name__ == "__main__":
    app()
