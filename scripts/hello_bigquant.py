"""BigQuant SDK smoke test — verify auth + data query + backtest work."""

from bigquant import dai, bigtrader


def main() -> None:
    print("==> Checking auth status...")
    try:
        user = __import__("bigquant").whoami()
        print(f"    Authenticated as: {user}")
    except Exception as e:
        print(f"    [FAIL] Not authenticated. Run:  bq auth configure")
        print(f"    Error: {e}")
        return

    print("\n==> Testing DAI data query...")
    df = dai.query(
        "SELECT date, instrument, close FROM cn_stock_bar1d WHERE date >= '2024-01-01' LIMIT 5",
        filters={"date": ["2024-01-01", "2024-12-31"]},
    ).pl()
    print(f"    Polars DataFrame ({len(df)} rows):")
    print(df)

    print("\n==> Testing BigTrader backtest...")

    def initialize(context):
        context.set_commission(bigtrader.PerOrder(buy_cost=0.0003, sell_cost=0.0003))
        context.asset = "000001.SZ"

    def handle_data(context, data):
        if context.get_account_position(context.asset).amount == 0:
            context.order_target_percent(context.asset, 1.0)

    perf = bigtrader.run(
        start_date="2024-01-01",
        end_date="2024-01-10",
        initialize=initialize,
        handle_data=handle_data,
        capital_base=1_000_000,
    )
    print(f"    Sharpe: {perf.summary['sharp_ratio']:.2f}")
    print("    [OK] All checks passed.")


if __name__ == "__main__":
    main()
