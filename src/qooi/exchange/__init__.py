"""OKX exchange adapter.

Public market data (no API key needed):
    >>> from qooi.exchange.market import MarketData
    >>> md = MarketData()
    >>> df = md.candles("BTC-USDT", bar="1H", limit=50)
    >>> df

Trading (API key required, see ``scripts/trade_okx.py``):
    >>> from qooi.exchange.trading import TradingClient
    >>> tc = TradingClient(api_key="...", secret_key="...", passphrase="...")
    >>> tc.place_order("BTC-USDT", side="buy", sz="0.01")
"""

from qooi.exchange.market import MarketData

__all__ = ["MarketData"]
