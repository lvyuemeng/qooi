"""Demo: OKX trading client — account, positions, place/cancel orders.

Before running, set your API credentials:

    # Windows (PowerShell):
    $env:OKX_API_KEY="your-api-key"
    $env:OKX_SECRET_KEY="your-secret-key"
    $env:OKX_PASSPHRASE="your-passphrase"

    # Or create a .env file in project root:
    OKX_API_KEY=your-api-key
    OKX_SECRET_KEY=your-secret-key
    OKX_PASSPHRASE=your-passphrase

Then run:
    uv run python scripts/trade_okx.py
"""

from qooi.exchange import TradingClient


def main() -> None:
    tc = TradingClient(flag="1")  # "1" = demo/testnet, "0" = live

    print("=== Account Balance ===")
    try:
        bal = tc.balance()
        print(bal.select(["ccy", "eq", "availBal", "frozenBal"]))
    except Exception as e:
        print(f"Balance error: {e}")

    print("\n=== Current Positions ===")
    try:
        pos = tc.positions()
        if pos.is_empty():
            print("No open positions.")
        else:
            print(pos)
    except Exception as e:
        print(f"Positions error: {e}")

    print("\n=== Place a market buy (0.001 BTC-USDT) ===")
    try:
        order = tc.place_order("BTC-USDT", side="buy", ord_type="market", sz="0.001")
        print("Order placed:", order)
        ord_id = order.get("ordId", "")
        if ord_id:
            print("\n=== Order status ===")
            status = tc.get_order("BTC-USDT", ord_id=ord_id)
            print(status)
    except Exception as e:
        print(f"Order error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
