import joblib
import typer
import pandas as pd
import webbrowser

from pathlib import Path
from loguru import logger
from lime.lime_tabular import LimeTabularExplainer

from bookwiseai.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

app = typer.Typer()

@app.command()
def main(
        test_path:     Path = PROCESSED_DATA_DIR / "test.csv",
        features_path: Path = PROCESSED_DATA_DIR / "features_voted.pkl",
        model_path:    Path = MODELS_DIR / "trained_models" / "voted_best_model.pkl",
        plot_path:     Path = FIGURES_DIR,
):
    logger.info("Running instance-level explanation of LIME method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]

    instance = X.iloc[0]
    logger.info(f"Instance to explain:\n{instance}\n")

    exp = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["0", "1"],
        discretize_continuous=False,
        verbose=True,
    )

    def predict_proba_with_names(arr):
        arr_df = pd.DataFrame(arr, columns=X.columns)
        return model.predict_proba(arr_df)

    lime = exp.explain_instance(
        data_row=instance.values,
        predict_fn=predict_proba_with_names,
    )

    OUT = plot_path / "lime_explanation.html"
    lime.save_to_file(OUT)
    logger.info(f"Saved LIME explanation to: {OUT}")
    webbrowser.open_new_tab(OUT.resolve().as_uri())

    logger.success("Instance-level explanation of Shapley method complete.")


if __name__ == "__main__":
    app()
