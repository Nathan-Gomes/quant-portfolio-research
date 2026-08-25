from __future__ import annotations

import json
import sqlite3
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fetch_yahoo_prices(config: dict, output_path: Path) -> pd.DataFrame:
    start = int(pd.Timestamp(config["data_start"], tz="UTC").timestamp())
    end = int((pd.Timestamp(config["data_end"], tz="UTC") + pd.Timedelta(days=1)).timestamp())
    frames: list[pd.DataFrame] = []

    for ticker, asset_class in config["assets"].items():
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
            result = json.load(response)["chart"]["result"][0]

        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize(),
                "ticker": ticker,
                "asset_class": asset_class,
                "adjusted_close": adjusted,
                "volume": quote["volume"],
            }
        ).dropna(subset=["adjusted_close"])
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False, float_format="%.6f")
    return prices


def load_price_matrix(csv_path: Path) -> pd.DataFrame:
    long_prices = pd.read_csv(csv_path, parse_dates=["date"])
    matrix = long_prices.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    return matrix.ffill().dropna()


def build_database(csv_path: Path, database_path: Path) -> None:
    prices = pd.read_csv(csv_path, parse_dates=["date"])
    assets = prices[["ticker", "asset_class"]].drop_duplicates().sort_values("ticker")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        assets.to_sql("assets", connection, if_exists="replace", index=False)
        prices.assign(date=prices["date"].dt.strftime("%Y-%m-%d")).to_sql(
            "daily_prices", connection, if_exists="replace", index=False
        )
        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_date_ticker
            ON daily_prices(date, ticker);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_ticker
            ON assets(ticker);
            """
        )


def run_sql_analysis(database_path: Path, query_path: Path) -> pd.DataFrame:
    query = query_path.read_text(encoding="utf-8")
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection)
