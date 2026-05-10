"""Apply market.py refactor: lazy CCXT, protocol registry, remove _days_since."""
import textwrap

path = r"src\qooi\exchange\market.py"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()
print(f"Read {len(c)} bytes")

# 1. Lazy CCXT: replace CcxtBackend.__init__
old_ccxt_init = """        self._ex = klass(config)
        try:
            self._ex.load_markets()
        except Exception as e:
            msg = f"Cannot connect to {exchange_id}"
            raise ConnectionError(msg + (f" via proxy {proxy}" if proxy else "")) from e

    def fetch_ohlcv"""
new_ccxt_init = """        self._ex = klass(config)
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        try:
            self._ex.load_markets()
            self._markets_loaded = True
        except Exception as e:
            msg = f"Cannot connect to {self._exchange_id}"
            raise ConnectionError(msg + (f" via proxy {self._proxy}" if self._proxy else "")) from e

    def fetch_ohlcv"""

if old_ccxt_init in c:
    c = c.replace(old_ccxt_init, new_ccxt_init)
    print("1. Lazy CCXT: applied")
else:
    print("1. Lazy CCXT: NOT FOUND")

# Add _ensure_markets() call to fetch_ohlcv and fetch_order_book
c = c.replace(
    "    def fetch_ohlcv(\n        self, symbol: str, timeframe: str = \"1d\", limit: int = 500, since: int | None = None\n    ) -> list[list]:\n        return self._ex.fetch_ohlcv",
    "    def fetch_ohlcv(\n        self, symbol: str, timeframe: str = \"1d\", limit: int = 500, since: int | None = None\n    ) -> list[list]:\n        self._ensure_markets()\n        return self._ex.fetch_ohlcv"
)
c = c.replace(
    "    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:\n        raw = self._ex.fetch_order_book",
    "    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:\n        self._ensure_markets()\n        raw = self._ex.fetch_order_book"
)

# 2. Replace ExchangeBackend with Protocols
new_protocols = textwrap.dedent("""\
# ---------------------------------------------------------------------------
# Protocols -- contracts for backend providers
# ---------------------------------------------------------------------------


class OhlcvProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None) -> list[list]: ...
    def fetch_ohlcv_range(self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000) -> pl.DataFrame: ...


class OrderBookProvider(Protocol):
    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot: ...
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None) -> list[list]: ...


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...


class FundingRateProvider(Protocol):
    def funding_rate_history(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame: ...
""")

# Replace the ExchangeBackend class comment block with protocols
old_class_start = """class ExchangeBackend:
    \"\"\"Synchronous exchange backend -- REST only."""

new_class_start = """class ExchangeBackend:
    \"\"\"Synchronous exchange backend -- REST only.  Prefer Protocols for typing."""

if old_class_start in c:
    # Insert protocols before ExchangeBackend class
    c = c.replace(old_class_start, new_protocols + "\n" + new_class_start)
    print("2. Protocols: inserted")
else:
    print("2. Protocols: NOT FOUND")

# 3. Remove _days_since helper
c = c.replace("""def _days_since(since: str) -> int:
    return max(
        1, (datetime.now(UTC) - datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)).days
    )


""", "")
print("3. _days_since: removed" if "_days_since" not in c else "3. _days_since: still present")

# 4. Add registry to MarketData.__init__
old_md_init = """    def __init__(self, exchange_id: str = "okx", proxy: str | None = None) -> None:
        self._backend: ExchangeBackend
        self._async_backend: CcxtProBackend | None = None
        self._proxy = proxy
        if exchange_id == "okx":
            self._backend = OkxSdkBackend(proxy)
        else:
            self._backend = CcxtBackend(exchange_id, proxy)
        self._exchange_id = exchange_id"""

new_md_init = """    _registry: dict[str, type[ExchangeBackend]] = {
        "okx": OkxSdkBackend,
    }
    _fallback: type[ExchangeBackend] = CcxtBackend

    def __init__(self, exchange_id: str = "okx", proxy: str | None = None) -> None:
        self._backend: ExchangeBackend
        self._async_backend: CcxtProBackend | None = None
        self._proxy = proxy
        backend_cls = self._registry.get(exchange_id, self._fallback)
        if backend_cls is OkxSdkBackend:
            self._backend = OkxSdkBackend(proxy, order_book=CcxtBackend("okx", proxy))
        else:
            self._backend = backend_cls(exchange_id, proxy)
        self._exchange_id = exchange_id"""

if old_md_init in c:
    c = c.replace(old_md_init, new_md_init)
    print("4. Registry: applied")
else:
    print("4. Registry: NOT FOUND")

# 5. Update OkxSdkBackend to accept order_book fallback via composition
old_okx_init = """    def __init__(self, proxy: str | None = None) -> None:
        super().__init__("okx", proxy)
        from okx.MarketData import MarketAPI

        self._api = MarketAPI(flag="1", debug=False)
        self._ccxt: CcxtBackend | None = None  # lazy init -- only needed for OB

    def _ensure_ccxt(self) -> CcxtBackend:
        if self._ccxt is None:
            self._ccxt = CcxtBackend("okx", self._proxy)
        return self._ccxt"""

new_okx_init = """    def __init__(self, proxy: str | None = None, *, order_book: OrderBookProvider | None = None) -> None:
        super().__init__("okx", proxy)
        from okx.MarketData import MarketAPI

        self._api = MarketAPI(flag="1", debug=False)
        self._ob_fallback: OrderBookProvider | None = order_book"""

if old_okx_init in c:
    c = c.replace(old_okx_init, new_okx_init)
    print("5a. OkxSdkBackend init: applied")
else:
    print("5a. OkxSdkBackend init: NOT FOUND")

# Replace _ensure_ccxt calls with _ob_fallback
c = c.replace("self._ensure_ccxt().fetch_ohlcv", "self._ob_fallback.fetch_ohlcv if self._ob_fallback else (_ for _ in ()).throw(RuntimeError('no fallback'))")
c = c.replace("return self._ensure_ccxt().fetch_order_book(symbol, limit)", "if self._ob_fallback:\n            return self._ob_fallback.fetch_order_book(symbol, limit)\n        raise RuntimeError(f\"OKX SDK order book failed for {symbol}, no fallback\")")

# Fix close()
c = c.replace(
    "    def close(self) -> None:\n        if self._ccxt:\n            self._ccxt.close()",
    "    def close(self) -> None:\n        if self._ob_fallback is not None:\n            try:\n                self._ob_fallback.close()\n            except AttributeError:\n                pass"
)
print("5b. OkxSdkBackend refactor: applied")

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print(f"Written {len(c)} bytes")
print(f"Contains _days_since: {'_days_since' in c}")
print(f"Contains _registry: {'_registry' in c}")
print(f"Contains _markets_loaded: {'_markets_loaded' in c}")
