"""Unified exchange adapter — OKX SDK (native) or CCXT (100+ exchanges).

Auto-selects backend::

    # OKX SDK (default) — ``inst_id`` like "BTC-USDT", bar like "1D"
    md = MarketData()
    df = md.candles("BTC-USDT", bar="1D", limit=100)

    # CCXT backend — ``symbol`` like "BTC/USDT", timeframe like "1d"
    md = MarketData(exchange_id="lbank")
    df = md.candles("BTC/USDT", bar="1d", limit=100)

    # Deep fetch with pagination
    df = md.candles_range("BTC/USDT", bar="1d", since="2018-01-01", limit=3000)

All methods return the same schema: timestamp, datetime, open, high, low, close, vol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
from okx.MarketData import MarketAPI


class MarketData:
    """Unified market data — OKX SDK or CCXT, same output schema."""

    def __init__(
        self,
        exchange_id: str = "okx",
        flag: str = "1",
        proxy: str | None = None,
    ) -> None:
        self._is_ccxt = exchange_id != "okx"
        if self._is_ccxt:
            self._init_ccxt(exchange_id, proxy)
        else:
            self._api = MarketAPI(flag=flag, debug=False)
        self._exchange_id = exchange_id

    def _init_ccxt(self, exchange_id: str, proxy: str | None) -> None:
        import ccxt

        klass = getattr(ccxt, exchange_id)
        config: dict[str, Any] = {"enableRateLimit": True}
        if proxy:
            config["proxies"] = {"https": proxy, "http": proxy}
        self._ex = klass(config)
        try:
            self._ex.load_markets()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to {exchange_id}" + (f" via {proxy}" if proxy else "")
            ) from e

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def candles(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        after: str | None = None,
        before: str | None = None,
    ) -> pl.DataFrame:
        if self._is_ccxt:
            return self._ccxt_ohlcv(inst_id, bar, limit=limit)
        params: dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self._okx_parse(self._api.get_candlesticks(**params))

    def candles_history(
        self,
        inst_id: str,
        bar: str = "1D",
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        if self._is_ccxt:
            return pl.DataFrame()  # CCXT uses candles_range for pagination
        params: dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self._okx_parse(self._api.get_history_candlesticks(**params))

    def candles_range(
        self,
        symbol: str,
        bar: str = "1d",
        since: str = "2020-01-01",
        limit: int = 3000,
    ) -> pl.DataFrame:
        """Paginaged OHLCV. Works with CCXT (deep history) or OKX SDK.

        For OKX backend, delegates to ``candles_history`` + ``candles``.
        For CCXT backend, uses the exchange's native pagination via ``since``.
        """
        if self._is_ccxt:
            return self._ccxt_paginate(symbol, bar, since, limit)
        # OKX backend: use history + recent
        cs = _okx_cache = __import__("qooi.exchange.store", fromlist=["CacheStore"]).CacheStore(
            md=self
        )
        return cs.refresh(symbol, bar=bar, days=_days_since(since), min_bars=limit)

    def _ccxt_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pl.DataFrame:
        raw = self._ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return self._parse_ohlcv(raw)

    def _ccxt_paginate(self, symbol: str, timeframe: str, since: str, limit: int) -> pl.DataFrame:
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        all_rows: list[list] = []
        while len(all_rows) < limit:
            raw = self._ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=500)
            if not raw:
                break
            all_rows.extend(raw)
            since_ms = raw[-1][0] + 1
            if len(raw) < 500:
                break
        if len(all_rows) > limit:
            all_rows = all_rows[:limit]
        return self._parse_ohlcv(all_rows)

    # ------------------------------------------------------------------
    # Funding rate (OKX only)
    # ------------------------------------------------------------------

    def funding_rate(self, inst_id: str = "BTC-USDT-SWAP") -> dict:
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = pub.get_funding_rate(instId=inst_id)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        return resp["data"][0] if resp.get("data") else {}

    def funding_rate_history(
        self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> pl.DataFrame:
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = pub.funding_rate_history(instId=inst_id, limit=str(limit))
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        rows = resp.get("data", [])
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "timestamp": int(r["fundingTime"]),
                    "funding_rate": float(r["realizedRate"]),
                    "funding_time": int(r["fundingTime"]),
                }
                for r in rows
            ]
        ).sort("timestamp")

    # ------------------------------------------------------------------
    # Order book (both backends)
    # ------------------------------------------------------------------

    def order_book(self, inst_id: str, sz: int = 5) -> dict[str, pl.DataFrame]:
        if self._is_ccxt:
            raw = self._ex.fetch_order_book(inst_id, limit=sz)

            def _side_ccxt(entries):
                if not entries:
                    return pl.DataFrame()
                return pl.DataFrame([{"price": float(e[0]), "size": float(e[1])} for e in entries])

            return {
                "asks": _side_ccxt(raw.get("asks", [])),
                "bids": _side_ccxt(raw.get("bids", [])),
            }
        resp = self._api.get_orderbook(instId=inst_id, sz=str(sz))
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        data = resp.get("data", [{}])[0]

        def _side(raw_side: list[list[str]]) -> pl.DataFrame:
            if not raw_side:
                return pl.DataFrame()
            return pl.DataFrame(
                [
                    {
                        "price": float(r[0]),
                        "size": float(r[1]),
                        "num_orders": int(r[2]),
                        "timestamp": int(r[3]),
                    }
                    for r in raw_side
                ]
            )

        return {"asks": _side(data.get("asks", [])), "bids": _side(data.get("bids", []))}

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _okx_parse(resp: dict) -> pl.DataFrame:
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        rows = resp.get("data", [])
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame(
            [
                {
                    "timestamp": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "vol": float(r[5]),
                    "vol_ccy": float(r[6]),
                    "vol_ccy_quote": float(r[7]),
                    "confirm": r[8],
                }
                for r in rows
            ]
        ).sort("timestamp")
        return df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))

    @staticmethod
    def _parse_ohlcv(raw: list[list]) -> pl.DataFrame:
        if not raw:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "timestamp": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "vol": float(r[5]),
                }
                for r in raw
            ]
        ).sort("timestamp")


def _days_since(since: str) -> int:
    return max(
        1, (datetime.now(UTC) - datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)).days
    )
