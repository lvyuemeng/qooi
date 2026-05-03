"""Demo: OKX trading client — account, positions, place/cancel orders.

Before running, set your API credentials in ``.env``:

    OKX_API_KEY=your-api-key
    OKX_SECRET_KEY=your-secret-key
    OKX_PASSPHRASE=your-passphrase
    OKX_FLAG=0        # 0 = live, 1 = testnet (use separate key)

Your current API key was created on the **live** environment.
To use testnet, create a separate key at https://www.okx.com/account/my-api
and set ``OKX_FLAG=1``.

Run:
    uv run python scripts/trade_okx.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

from qooi.exchange import TradingClient


def main() -> None:
    flag = os.getenv("OKX_FLAG", "1")
    print(f"Environment: {'LIVE' if flag == '0' else 'TESTNET'}")
    print()

    tc = TradingClient(flag=flag)

    print("=== Account Balance ===")
    try:
        bal = tc.balance()
        if bal.is_empty():
            print("(empty — no assets)")
        else:
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

    print("\n=== Place a market buy (10 USDT worth of SOL-USDT) ===")
    try:
        order = tc.place_order("SOL-USDT", side="buy", ord_type="market", sz="10")
        print("Order placed:", order)
        ord_id = order.get("ordId", "")
        if ord_id:
            print("\n=== Order status ===")
            status = tc.get_order("SOL-USDT", ord_id=ord_id)
            print(status)
    except Exception as e:
        print(f"Order error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
