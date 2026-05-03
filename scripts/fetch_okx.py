"""Demo: fetch OKX market data (no API key needed).

Usage:
    uv run python scripts/fetch_okx.py
"""

from qooi.exchange import MarketData


def main() -> None:
    md = MarketData(flag="1")  # demo/testnet

    print("=== BTC-USDT 1H candles (last 5) ===")
    df = md.candles("BTC-USDT", bar="1H", limit=5)
    print(df.select(["datetime", "open", "high", "low", "close", "vol"]))
    print()

    # === Also try with history endpoint to get more data ===
    print("=== BTC-USDT 1D history (last 3) ===")
    hist = md.candles_history("BTC-USDT", bar="1D", limit=3)
    print(hist.select(["datetime", "open", "close", "vol"]))

    print("\n=== BTC-USDT ticker ===")
    ticker = md.ticker("BTC-USDT")
    print(ticker)

    print("\n=== BTC-USDT order book (top 3) ===")
    ob = md.order_book("BTC-USDT", sz=3)
    print("Asks:", ob["asks"])
    print("Bids:", ob["bids"])

    print("\n=== All SPOT instruments (first 5) ===")
    inst = md.instruments(inst_type="SPOT")
    print(inst.select(["instId", "baseCcy", "quoteCcy", "state"]).head(5))


if __name__ == "__main__":
    main()
