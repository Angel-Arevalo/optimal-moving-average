"""Run through uv; use the official Quantiacs hourly crypto loader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from read_data import prepare_asset
from .market import validate_market


def download(asset="BTC", start="2021-12-27", end="2024-02-26", output=None):
    root = Path(__file__).resolve().parent
    os.environ.setdefault("CACHE_DIR", str(root / "cache" / "quantiacs"))
    # qnt explicitly supports API_KEY=default for public data. Respect a supplied key.
    os.environ.setdefault("API_KEY", "default")
    import qnt.data as qndata
    data = qndata.crypto.load_data(assets=[asset], min_date=start, max_date=end)
    frame = prepare_asset(data.sel(asset=asset))
    # End is exclusive. Source rows are hourly bar starts.
    frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
    validate_market(frame)
    output = Path(output) if output else root / "data" / f"{asset}_hourly.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    metadata = dict(provider="Quantiacs", loader="qnt.data.crypto.load_data", asset=asset,
                    requested_start=start, requested_end_exclusive=end, rows=len(frame),
                    first=str(frame.index[0]), last=str(frame.index[-1]),
                    timestamps="UTC naive, bar starts; available one hour later",
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--start", default="2021-12-27")
    parser.add_argument("--end", default="2024-02-26")
    parser.add_argument("--output", type=Path)
    download(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
