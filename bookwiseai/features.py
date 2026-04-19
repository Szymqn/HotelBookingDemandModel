import typer
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from bookwiseai.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

MONTH_MAP = {
    "January": 1,   "February": 2,  "March": 3,     "April": 4,
    "May": 5,       "June": 6,      "July": 7,      "August": 8,
    "September": 9, "October": 10,  "November": 11, "December": 12,
}

CAT_COLS = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "assigned_room_type", "deposit_type", "customer_type",
]

_EUROPE_ISO = {
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CYP", "CZE",
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "FRO", "GBR", "GEO", "GGY",
    "GIB", "GRC", "HRV", "HUN", "IMN", "IRL", "ISL", "ITA", "JEY", "LIE",
    "LTU", "LUX", "LVA", "MCO", "MKD", "MLT", "MNE", "NLD", "NOR", "POL",
    "ROU", "RUS", "SMR", "SRB", "SVK", "SVN", "SWE", "UKR",
}

SEASON_MAP = {
    12: "winter", 1: "winter",  2: "winter",
    3:  "spring", 4: "spring",  5: "spring",
    6:  "summer", 7: "summer",  8: "summer",
    9:  "autumn", 10: "autumn", 11: "autumn",
}

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]

LEAD_TIME_BUCKET_ORDER = ["last_minute", "short", "medium", "long", "very_long", "extreme"]

FINAL_FEATURES = [
    # Numeric
    "lead_time", "arrival_date_year", "arrival_month_number",
    "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr_clipped", "required_car_parking_spaces", "total_of_special_requests",
    "total_nights", "total_guests", "is_weekend_arrival",
    "has_previous_cancellations", "is_loyal_guest", "long_lead_time",
    "has_agent", "has_company", "month_sin", "month_cos",
    # Encoded categorical
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
    "assigned_room_type", "season", "lead_time_bucket", "country_region",
]

class HotelFeaturesEngineer:
    def __init__(self):
        self.top_rooms = []

        self.room_encoder = None

        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

        self.ordered_encoder = OrdinalEncoder(
            categories=[SEASON_ORDER, LEAD_TIME_BUCKET_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    @staticmethod
    def _apply_safe_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["arrival_month_number"] = df["arrival_date_month"].map(MONTH_MAP)
        df["children"] = df["children"].fillna(0)
        df["meal"] = df["meal"].replace("Undefined", "SC")
        df["country"] = df["country"].fillna("N/A")

        df["has_agent"] = df["agent"].notna().astype(int)
        df["has_company"] = df["company"].notna().astype(int)

        df["adr_clipped"] = df["adr"].clip(lower=0, upper=1000)

        def _region_score(c: str) -> int:
            if c == "PRT": return 1
            if c in _EUROPE_ISO: return 2
            return 3

        df["country_region"] = df["country"].astype(str).apply(_region_score)

        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]
        df["season"] = df["arrival_month_number"].map(SEASON_MAP)
        df["is_weekend_arrival"] = (df["stays_in_weekend_nights"] > 0).astype(int)
        df["has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
        df["is_loyal_guest"] = (df["previous_bookings_not_canceled"] > 0).astype(int)
        df["long_lead_time"] = (df["lead_time"] > 90).astype(int)
        df["lead_time_bucket"] = pd.cut(
            df["lead_time"],
            bins=[-1, 7, 30, 90, 180, 365, 9999],
            labels=LEAD_TIME_BUCKET_ORDER,
        ).astype(str)

        df["month_sin"] = np.sin(2 * np.pi * df["arrival_month_number"] / 12).astype("float32")
        df["month_cos"] = np.cos(2 * np.pi * df["arrival_month_number"] / 12).astype("float32")

        return df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X = self._apply_safe_features(df=X_train)

        self.top_rooms = X["reserved_room_type"].value_counts().head(5).index.tolist()

        X["reserved_room_type"] = X["reserved_room_type"].apply(
            lambda v: v if v in self.top_rooms else "Other"
        )

        self.room_encoder = OrdinalEncoder(
            categories=[self.top_rooms + ["Other"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X["reserved_room_type"] = self.room_encoder.fit_transform(X[["reserved_room_type"]])

        X[CAT_COLS] = self.ordinal_encoder.fit_transform(X[CAT_COLS].astype(str))

        self.ordered_encoder.fit(X[["season", "lead_time_bucket"]])
        X[["season", "lead_time_bucket"]] = self.ordered_encoder.transform(
            X[["season", "lead_time_bucket"]]
        )

        return X[FINAL_FEATURES]

    def transform(self, X_eval: pd.DataFrame) -> pd.DataFrame:
        X = self._apply_safe_features(df=X_eval)

        X["reserved_room_type"] = X["reserved_room_type"].apply(
            lambda v: v if v in self.top_rooms else "Other"
        )
        X["reserved_room_type"] = self.room_encoder.transform(X[["reserved_room_type"]])

        X[CAT_COLS] = self.ordinal_encoder.transform(X[CAT_COLS].astype(str))

        X[["season", "lead_time_bucket"]] = self.ordered_encoder.transform(
            X[["season", "lead_time_bucket"]]
        )

        return X[FINAL_FEATURES]

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "hotel_bookings.csv",
    interim_path: Path = INTERIM_DATA_DIR / "hotel_bookings_clean.csv",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    valid_path: Path = PROCESSED_DATA_DIR / "valid.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
):
    logger.info("Generating features from dataset...")

    df = pd.read_csv(filepath_or_buffer=input_path)

    TARGET = "is_canceled"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    engineer = HotelFeaturesEngineer()

    X_train = engineer.fit_transform(X_train)
    X_val = engineer.transform(X_val)
    X_test = engineer.transform(X_test)

    X_train.to_csv(interim_path, index=False)
    logger.info(f"Interim saved (train only, without target) to {interim_path}")

    joblib.dump(engineer, PROCESSED_DATA_DIR / "engineer.pkl")
    logger.info(f"Engineer saved to {PROCESSED_DATA_DIR}/engineer.pkl")

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    valid = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Train : {train.shape} | cancellation rate: {y_train.mean():.2%}")
    logger.info(f"Val   : {valid.shape} | cancellation rate: {y_val.mean():.2%}")
    logger.info(f"Test  : {test.shape}  | cancellation rate: {y_test.mean():.2%}")

    logger.success("Features generation complete.")

if __name__ == "__main__":
    app()
