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
    logger.info("Running dataset-level explanation of variable importance measure method...")

    test = pd.read_csv(test_path)
    features = joblib.load(features_path)
    model = joblib.load(model_path)

    X = test[features]
    y = test['is_canceled']

    exp = dx.Explainer(model, X, y, label="FS Voted, RF Model Pipeline")

    logger.info("Generating variable importance measure explanation for dataset...")
    mp = exp.model_parts()
    logger.info(f"\n{mp.result}")
    mp.plot(show=True)

    fig = mp.plot(show=False)
    OUT_path = plot_path / "vim.png"
    fig.write_image(OUT_path)
    logger.info(f"Saved vim plot to: {OUT_path}")

    vi_grouped = exp.model_parts(
        variable_groups={
            'lead_time_metrics': [
                'lead_time',
                'long_lead_time',
                'lead_time_bucket'
            ],
            'seasonality_and_dates': [
                'arrival_date_year',
                'arrival_month_number',
                'arrival_date_week_number',
                'month_cos'
            ],
            'stay_duration': [
                'stays_in_weekend_nights',
                'stays_in_week_nights',
                'total_nights'
            ],
            'guest_history_and_loyalty': [
                'previous_cancellations',
                'previous_bookings_not_canceled',
                'has_previous_cancellations',
                'is_loyal_guest'
            ],
            'guest_profile': [
                'total_guests',
                'customer_type',
                'country_region'
            ],
            'room_and_property': [
                'hotel',
                'reserved_room_type',
                'assigned_room_type',
                'booking_changes'
            ],
            'amenities_and_requests': [
                'meal',
                'required_car_parking_spaces',
                'total_of_special_requests'
            ],
            'financials': [
                'adr_clipped',
                'deposit_type'
            ],
            'sourcing_and_marketing': [
                'market_segment',
                'distribution_channel',
                'has_agent'
            ]
        }
    )

    logger.info(f"\n{vi_grouped.result}")
    vi_grouped.plot(show=True)

    fig_grouped = vi_grouped.plot(show=False)
    OUT_grouped_path = plot_path / "vim_grouped.png"
    fig_grouped.write_image(OUT_grouped_path)
    logger.info(f"Saved vim grouped plot to: {OUT_grouped_path}")

    logger.success("Dataset-level explanation of variable importance measure method complete.")


if __name__ == "__main__":
    app()
