import typer
import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
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
    "reserved_room_type", "deposit_type", "customer_type",
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

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Drop Leaky Columns
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])

    # Impute Missing Values
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["children"]          = df["children"].fillna(0)
    df["meal"]              = df["meal"].replace("Undefined", "SC")
    df["country"]           = df["country"].fillna("N/A")

    # Binary indicators for presence of agent/company (instead of IDs)
    df["has_agent"]   = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)

    df = df.drop(columns=["agent", "company"])

    # eliminate negative ADR and cap at 1000 (outliers)
    df["adr_clipped"] = df["adr"].clip(lower=0, upper=1000)

    def _region_score(c: str) -> str:
        match c:
            case "PRT": return 1
            case _  if c in _EUROPE_ISO: return 2
            case _: return 3

    df["country_region"] = df["country"].astype(str).map(_region_score)
    df = df.drop(columns=["country"])

    top_rooms = df["reserved_room_type"].value_counts().head(5).index.tolist()
    for col in ["reserved_room_type"]:
        df[col] = df[col].apply(lambda v: v if v in top_rooms else "Other")

    df["total_nights"]               = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"]               = df["adults"] + df["children"] + df["babies"]
    df["season"]                     = df["arrival_month_num"].map(SEASON_MAP)
    df["is_weekend_arrival"]         = (df["stays_in_weekend_nights"] > 0).astype(int)
    df["has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
    df["is_loyal_guest"]             = (df["previous_bookings_not_canceled"] > 0).astype(int)
    df["long_lead_time"]             = (df["lead_time"] > 90).astype(int)
    df["lead_time_bucket"]           = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, 9999],
        labels=["last_minute", "short", "medium", "long", "very_long", "extreme"],
    ).astype(str)

    # cyclic encoding of months (for Logistic Regression)
    df["month_sin"] = np.sin(2 * np.pi * df["arrival_month_num"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["arrival_month_num"] / 12).astype("float32")

    # Label Encoding for categorical features (for tree-based models)
    le = LabelEncoder()
    encode_cols = CAT_COLS + ["season", "lead_time_bucket", "country_region"]
    for col in encode_cols:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Ordinal Encoding for reserved_room_type (for Logistic Regression)
    if "reserved_room_type" in df.columns:
        oe = OrdinalEncoder(categories=[top_rooms + ["Other"]])
        df["reserved_room_type_enc"] = oe.fit_transform(df[["reserved_room_type"]])

    return df

def build_feature_matrix(data: pd.DataFrame):
    df = data.copy()

    numeric_features = [
        "is_canceled", "lead_time", "arrival_date_year", "arrival_month_num",
        "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
        "adr_clipped", "required_car_parking_spaces", "total_of_special_requests",
        "total_nights", "total_guests", "is_weekend_arrival",
        "has_previous_cancellations", "is_loyal_guest", "long_lead_time",
        "has_agent", "has_company",
        "month_sin", "month_cos",
    ]

    encoded_features = [
        "hotel_enc", "meal_enc", "market_segment_enc", "distribution_channel_enc",
        "reserved_room_type_enc", "deposit_type_enc", "customer_type_enc",
        "season_enc", "lead_time_bucket_enc", "country_region_enc",
    ]

    for enc_col in encoded_features:
        original_col = enc_col.replace("_enc", "")
        df[original_col] = df[enc_col]  # overwrite original with encoded values
        df = df.drop(columns=[enc_col])  # drop the _enc column

    renamed_encoded = [col.replace("_enc", "") for col in encoded_features]

    all_features = numeric_features + renamed_encoded
    return df[all_features]

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

    df_ef = engineer_features(df)
    df_clean = build_feature_matrix(df_ef)
    logger.info(f"Feature matrix shape: {df_clean.shape}")
    df_clean.to_csv(interim_path, index=False)
    logger.info("Feature engineering complete")

    TARGET = "is_canceled"
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

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

    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Train : {train.shape}")
    logger.info(f"Val   : {valid.shape}")
    logger.info(f"Test  : {test.shape}")

    logger.info(f"Train : {train.shape} | cancellation rate: {y_train.mean():.2%}")
    logger.info(f"Val   : {valid.shape} | cancellation rate: {y_val.mean():.2%}")
    logger.info(f"Test  : {test.shape}  | cancellation rate: {y_test.mean():.2%}")

    logger.success("Features generation complete.")

if __name__ == "__main__":
    app()
