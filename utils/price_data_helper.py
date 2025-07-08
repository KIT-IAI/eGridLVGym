import pandas as pd
import numpy as np


def prepare_price_data():
    """
    Extract the year 2023 from the spotmarket data, drop unused collumns and save it to a new csv file.
    """
    data = pd.read_csv("./data/spotmarket.csv", index_col=0, parse_dates=True, sep=";")

    data.index = data.index + " " + data["von"]
    data.index
    data.index = pd.to_datetime(data.index, format="%d.%m.%Y %H:%M")
    data["price"] = data["Spotmarktpreis in ct/kWh"]
    data = data.drop(
        ["von", "Spotmarktpreis in ct/kWh", "bis", "Zeitzone von", "Zeitzone bis"],
        axis=1,
    )

    data = data[~data.index.duplicated(keep="first")]
    data["price"] = data["price"].str.replace(",", ".").astype(float)
    data = data.resample("15min").interpolate(method="pad")

    data = data["2023-01-01":"2023-12-31"]
    with open("./data/spotmarket_reduced_quarters_2023.csv", "w") as f:
        data.to_csv(f)


def clip_and_normalize_price():
    """
    Clip the price data to the 15% and 85% quantile and normalize it (min-max normalization).
    """
    data = pd.read_csv(
        "./data/spotmarket_reduced_quarters_2023.csv", index_col=0, parse_dates=True
    )
    q1 = data["price"].quantile(0.15)
    q3 = data["price"].quantile(0.85)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    data["price_clipped"] = np.clip(data["price"], lower_bound, upper_bound)
    data["price_clipped_normalized"] = (
        data["price_clipped"] / data["price_clipped"].abs().max()
    )

    data["price_normalized"] = data["price"] / data["price"].abs().max()

    with open("./data/spotmarket_reduced_quarters_2023.csv", "w") as f:
        data.to_csv(f)


if __name__ == "__main__":
    prepare_price_data()
    clip_and_normalize_price()
