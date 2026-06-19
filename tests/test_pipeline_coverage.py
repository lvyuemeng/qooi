from __future__ import annotations

import polars as pl

from qooi.pipeline import HOUR_MS, now_ms
from qooi.pipeline.coverage import CoverageRunPolicy, plan_product_coverage, source_spec


def test_provider_bounded_stale_source_gets_latest_refresh_job() -> None:
    symbol = "BTC-USDT-SWAP"
    stale_ts = now_ms() - 3 * HOUR_MS
    frame = pl.DataFrame({"symbol": [symbol], "timestamp": [stale_ts]})
    spec = source_spec(
        product_name="open_interest",
        target_days=1,
        max_staleness_hours=2,
        page_limit=100,
    )

    plan = plan_product_coverage(
        spec=spec,
        symbols=(symbol,),
        frame=frame,
        coin_listed_ms={},
        policy=CoverageRunPolicy(max_requests=10),
        provider_bounded={(symbol, "open_interest", "1H")},
    )

    assert plan.states[0].status == "stale"
    assert plan.jobs[0].kind == "latest_refresh"
