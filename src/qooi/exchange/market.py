"""OKX exchange adapter — public market data (no API key needed)."""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl
from okx.MarketData import MarketAPI


class MarketData:
    """OKX public market data — no authentication required.

    Usage::

        md = MarketData()
        df = md.candles("BTC-USDT", bar="1H", limit=100)
        ticker = md.ticker("BTC-USDT")
    """

    OKX_BASE = "https://www.okx.com"

    def __init__(self, flag: str = "1") -> None:
        self._api = MarketAPI(flag=flag, debug=False)
        self._http = httpx.Client(base_url=self.OKX_BASE, timeout=15)

    def _get(self, path: str, params: dict | None = None) -> dict:
        data = self._http.get(path, params=params).json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        return data

    # ------------------------------------------------------------------
    # K-line / Candlesticks
    # ------------------------------------------------------------------

    def candles(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        after: str | None = None,
        before: str | None = None,
    ) -> pl.DataFrame:
        """Fetch historical candlestick data.

        Returns columns: timestamp, datetime, open, high, low, close, vol, ...
        """
        params: dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        resp = self._api.get_candlesticks(**params)
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

    def candles_history(
        self,
        inst_id: str,
        bar: str = "1D",
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """Fetch historical candlestick data (archived, up to 3 months)."""
        params: dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        resp = self._api.get_history_candlesticks(**params)
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

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    def ticker(self, inst_id: str) -> pl.DataFrame:
        resp = self._api.get_ticker(instId=inst_id)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        return pl.DataFrame(resp.get("data", []))

    def tickers(self, inst_type: str = "SPOT") -> pl.DataFrame:
        resp = self._api.get_tickers(instType=inst_type)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        return pl.DataFrame(resp.get("data", []))

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def order_book(self, inst_id: str, sz: int = 5) -> dict[str, pl.DataFrame]:
        resp = self._api.get_orderbook(instId=inst_id, sz=str(sz))
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX API error: {resp.get('msg', resp)}")
        data = resp.get("data", [{}])[0]

        def _side(raw: list[list[str]]) -> pl.DataFrame:
            if not raw:
                return pl.DataFrame()
            return pl.DataFrame(
                [
                    {
                        "price": float(r[0]),
                        "size": float(r[1]),
                        "num_orders": int(r[2]),
                        "timestamp": int(r[3]),
                    }
                    for r in raw
                ]
            )

        return {
            "asks": _side(data.get("asks", [])),
            "bids": _side(data.get("bids", [])),
        }

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def instruments(self, inst_type: str = "SPOT") -> pl.DataFrame:
        data = self._get("/api/v5/public/instruments", {"instType": inst_type})
        return pl.DataFrame(data.get("data", []))
