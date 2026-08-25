from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_pipeline.data import fetch_yahoo_prices, load_config


if __name__ == "__main__":
    config = load_config(ROOT / "config" / "research.yaml")
    output = ROOT / "data" / "etf_prices.csv"
    prices = fetch_yahoo_prices(config, output)
    print(f"Wrote {len(prices):,} rows to {output}")
