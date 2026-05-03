"""Demo: fetch A-share daily bars with TickFlow free tier."""

from __future__ import annotations

from qooi.data import DataSource


def main() -> None:
    ds = DataSource()

    print("=== Single stock daily bars ===")
    df = ds.daily_bars("600000.SH", count=10)
    print(df)

    print("\n=== Batch daily bars ===")
    result = ds.daily_bars_batch(["600000.SH", "000001.SZ", "600519.SH"], count=5)
    for sym, frame in result.items():
        print(f"\n{sym}:")
        print(frame)

    print("\n=== Instrument info ===")
    info = ds.instruments(["600000.SH", "000001.SZ"])
    print(info)


if __name__ == "__main__":
    main()
