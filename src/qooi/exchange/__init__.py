"""OKX exchange adapter.

Public market data (no API key needed):
    >>> from qooi.exchange.market import MarketData
    >>> md = MarketData()
    >>> df = md.candles("BTC-USDT", bar="1H", limit=50)

Trading (API key via env vars OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE):
    >>> from qooi.exchange.trading import TradingClient
    >>> tc = TradingClient()
    >>> tc.balance()
"""

from qooi.exchange.market import (
    CcxtBackend,
    FundingRateProvider,
    MarketData,
    ObSnapshot,
    OhlcvProvider,
    OkxSdkBackend,
    OrderBookProvider,
    StreamProvider,
)

__all__ = [
    "SyncMarketData",
    "MarketData",
    "TradingClient",
    "ObSnapshot",
    "OhlcvProvider",
    "OrderBookProvider",
    "StreamProvider",
    "FundingRateProvider",
    "CcxtBackend",
    "OkxSdkBackend",
]
