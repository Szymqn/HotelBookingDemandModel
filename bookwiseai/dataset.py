import typer
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm import tqdm

from bookwiseai.config import RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "hotel_bookings.csv",
):
    logger.info("Processing dataset...")

    df = pd.read_csv(filepath_or_buffer=input_path)

    logger.info(f"Loaded: {df.shape[0]:,} reservation x {df.shape[1]} features")
    logger.info(f"Hotels: {df['hotel'].value_counts().to_dict()}")
    logger.info(f"Cancellations: {df['is_canceled'].value_counts().to_dict()}")

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
