"""State provider tests."""

from qooi.core.basket import Basket, BasketState
from qooi.core.config import PAIRS
from qooi.core.state import BacktestStateProvider, JsonSoftStateStore, OkxStateProvider


class MemorySoftStore:
    def __init__(self, data=None):
        self.data = data or {}
        self.written = None

    def read(self):
        return self.data

    def write(self, baskets):
        self.written = {b.basket_id: b.current_sz for b in baskets}


class FakeTradingClient:
    def positions(self, inst_type="SWAP"):
        return [{"instId": "ETH-USDT-SWAP", "pos": "2", "avgPx": "3000"}]

    def orders(self, inst_id, inst_type="SWAP"):
        return [{"side": "buy", "sz": "2", "px": "3001", "ordId": "o1"}]


def test_backtest_state_provider_is_memory_only():
    basket = Basket("x", "X", "s", "buy", state=BasketState.ACTIVE, current_sz=1.0)
    provider = BacktestStateProvider([basket])

    loaded = provider.load(PAIRS)
    assert loaded == [basket]

    provider.save_soft([])
    assert provider.baskets == []


def test_okx_state_provider_merges_hard_and_soft_state():
    pair = PAIRS[0]
    strategy_id = "momentum_burst"
    basket_id = f"{pair.asset.symbol}-{strategy_id}"
    soft = MemorySoftStore({basket_id: {"bars_in_pos": 7, "recovery_level": 1}})
    provider = OkxStateProvider(FakeTradingClient(), soft)

    baskets = provider.load([pair], strategy_id=strategy_id)
    basket = baskets[0]

    assert basket.is_active
    assert basket.side == "buy"
    assert basket.current_sz == 2.0
    assert basket.entry_px == 3000.0
    assert basket.bars_in_pos == 7
    assert basket.recovery_activated is True
    assert len(basket.positions) == 1


def test_json_soft_store_writes_only_soft_fields(tmp_path):
    path = tmp_path / "baskets.json"
    store = JsonSoftStateStore(path)
    basket = Basket(
        "x",
        "X",
        "s",
        "buy",
        state=BasketState.ACTIVE,
        entry_px=100.0,
        current_sz=3.0,
        bars_in_pos=2,
    )

    store.write([basket])
    data = store.read()["x"]

    assert data["bars_in_pos"] == 2
    assert "current_sz" not in data
    assert "entry_px" not in data
