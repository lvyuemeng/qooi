"""TickFlow data access layer — unified interface for A-share market data."""

from __future__ import annotations

import io
import sys
from collections.abc import Sequence
from unittest.mock import patch

import polars as pl
from tickflow import TickFlow


class DataSource:
    """Wrapper around TickFlow SDK for A-share data.

    Uses the free tier by default (daily K-line only, no auth needed).
    Pass ``api_key`` to unlock real-time / minute data.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            self._tf = TickFlow(api_key=api_key)
        else:
            with patch.object(sys, "stdout", io.StringIO()):  # suppress emoji banner
                self._tf = TickFlow.free()

    # ------------------------------------------------------------------
    # Daily bars
    # ------------------------------------------------------------------

    def daily_bars(self, symbol: str, count: int = 1000) -> pl.DataFrame:
        """Fetch daily K-line for a single A-share instrument.

        Parameters
        ----------
        symbol:
            Instrument code with market suffix, e.g. ``"600000.SH"``.
        count:
            Number of bars to fetch (max 10 000).

        Returns
        -------
        Polars DataFrame with columns: date, open, high, low, close, volume, …
        """
        df = self._tf.klines.get(symbol, period="1d", count=count, as_dataframe=True)
        return pl.from_pandas(df)

    def daily_bars_batch(
        self, symbols: Sequence[str], count: int = 1000
    ) -> dict[str, pl.DataFrame]:
        """Fetch daily K-line for multiple instruments.

        Returns a dict keyed by symbol.
        """
        raw = self._tf.klines.batch(
            list(symbols),
            period="1d",
            count=count,
            as_dataframe=True,
            show_progress=True,
        )
        return {sym: pl.from_pandas(df) for sym, df in raw.items()}

    # ------------------------------------------------------------------
    # Intraday (requires API key)
    # ------------------------------------------------------------------

    def intraday_bars(
        self, symbol: str, period: str = "1m", count: int | None = None
    ) -> pl.DataFrame:
        """Fetch intraday minute bars for the current trading day.

        Requires a paid API key. Free tier will raise an error.
        """
        df = self._tf.klines.intraday(symbol, period=period, count=count, as_dataframe=True)
        return pl.from_pandas(df)

    # ------------------------------------------------------------------
    # Real-time quotes (requires API key)
    # ------------------------------------------------------------------

    def quotes(
        self, symbols: Sequence[str] | None = None, universe: str | None = None
    ) -> pl.DataFrame:
        """Fetch real-time quotes.

        Pass either ``symbols`` (list of codes) or ``universe``
        (e.g. ``"CN_Equity_A"`` for all A-shares).
        """
        kwargs = {}
        if symbols:
            kwargs["symbols"] = list(symbols)
        if universe:
            kwargs["universes"] = [universe]
        quotes = self._tf.quotes.get(**kwargs, as_dataframe=True)
        return pl.from_pandas(quotes)

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def instruments(self, symbols: Sequence[str]) -> pl.DataFrame:
        """Query instrument metadata for the given symbols."""
        raw = self._tf.instruments.batch(list(symbols))
        return pl.from_pandas(raw) if hasattr(raw, "head") else pl.DataFrame(raw)
