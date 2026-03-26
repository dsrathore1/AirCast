import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def create_time_features(df):
    df["is_weekend"] = df["day"].apply(lambda x: 1 if x in [6, 7] else 0)

    df["season"] = df["month"] % 12 // 3 + 1
    # 1 = winter, 2 = spring, 3 = summer, 4 = monsoon-ish

    return df


def add_city_pollution_stats(df):
    city_avg = df.groupby("city")["pollutant_avg"].transform("mean")
    df["city_avg_pollution"] = city_avg

    return df


def add_lag_features(df):
    df = df.sort_values(by=["city", "year", "month", "day", "hour"])

    df["prev_pollution"] = df.groupby("city")["pollutant_avg"].shift(1)

    df = df.dropna()  # lag creates NaN
    return df


def select_features(df):
    features = [
        "city",
        "state",
        "pollutant_id",
        "year",
        "month",
        "day",
        "hour",
        "is_weekend",
        "season",
        "city_avg_pollution",
        "prev_pollution",
    ]

    target = "pollutant_avg"

    X = df[features]
    y = df[target]

    return X, y


def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Feature-engineered data saved to {path}")


def run_pipeline(input_path, output_path):
    df = load_data(input_path)

    df = create_time_features(df)
    df = add_city_pollution_stats(df)
    df = add_lag_features(df)

    save_data(df, output_path)


if __name__ == "__main__":
    run_pipeline("Dataset/clean_air_quality.csv", "Dataset/feature_data.csv")
