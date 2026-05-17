from __future__ import annotations

import polars as pl

from qooi.strategies.features import add_volatility_regime


def test_flow_feature_volatility_regime_is_synthetic_and_hermetic():
    frame = pl.DataFrame(
        {
            "close": [100.0 + idx * 0.1 for idx in range(220)],
        }
    )

    out = add_volatility_regime(short_span=4, long_span=12)(frame)

    assert "volatility_regime" in out.columns
    assert out.height == frame.height
