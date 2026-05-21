"""State provider tests."""

from qooi.core.basket import Basket, BasketState
from qooi.core.state import (
    BacktestStateProvider,
    BasketStateSource,
    EvaluatedBasketState,
    JsonSoftStateStore,
    OkxOrderSource,
    OkxPositionSource,
    OkxStateProvider,
    evaluate_basket_source,
    format_basket_id,
    format_okx_client_id,
    parse_basket_id,
)
from qooi.research.instruments import CORE_UNIVERSE


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
        basket_id = format_basket_id("ETH-USDT-SWAP", "momentum_burst")
        return [
            {
                "instId": "ETH-USDT-SWAP",
                "side": "buy",
                "sz": "2",
                "px": "3001",
                "ordId": "o1",
                "clOrdId": format_okx_client_id(basket_id),
            }
        ]


def test_backtest_state_provider_is_memory_only():
    basket = Basket("x", "X", "s", "buy", state=BasketState.ACTIVE, current_sz=1.0)
    provider = BacktestStateProvider([basket])

    loaded = provider.load(CORE_UNIVERSE)
    assert loaded == [basket]

    provider.save_soft([])
    assert provider.baskets == []


def test_okx_state_provider_reconstructs_from_okx_only():
    pair = CORE_UNIVERSE[0]
    strategy_id = "momentum_burst"
    provider = OkxStateProvider(FakeTradingClient())

    baskets = provider.load([pair], strategy_id=strategy_id)
    basket = baskets[0]

    assert basket.is_active
    assert basket.side == "buy"
    assert basket.current_sz == 2.0
    assert basket.entry_px == 3000.0
    assert basket.bars_in_pos == 0
    assert basket.recovery_activated is False
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


def test_okx_state_provider_ignores_local_state_by_default():
    class FlatTradingClient:
        def positions(self, inst_type="SWAP"):
            return []

        def orders(self, inst_id, inst_type="SWAP"):
            return []

    pair = CORE_UNIVERSE[0]
    strategy_id = "momentum_burst"
    provider = OkxStateProvider(FlatTradingClient())

    basket = provider.load([pair], strategy_id=strategy_id)[0]

    assert basket.is_idle
    assert basket.suspended_long is False
    assert basket.suspension_px == 0.0
    assert basket.recovery_activated is False


def test_okx_state_provider_reconstructs_branch_from_okx_client_id():
    class FlatTradingClient:
        def positions(self, inst_type="SWAP"):
            return []

        def orders(self, inst_id, inst_type="SWAP"):
            return []

    pair = CORE_UNIVERSE[0]
    strategy_id = "momentum_burst"
    hedge_id = format_basket_id(pair.asset.symbol, strategy_id, "hedge")
    class BranchTradingClient(FlatTradingClient):
        def orders(self, inst_id, inst_type="SWAP"):
            return [
                {
                    "instId": pair.asset.symbol,
                    "side": "sell",
                    "sz": "1",
                    "px": "99",
                    "ordId": "h1",
                    "clOrdId": format_okx_client_id(hedge_id),
                }
            ]

    provider = OkxStateProvider(BranchTradingClient())

    baskets = provider.load([pair], strategy_id=strategy_id)

    hedge = next(b for b in baskets if b.basket_id == hedge_id)
    assert hedge.is_idle
    assert len(hedge.positions) == 1


def test_evaluate_basket_source_returns_data_without_mutating_basket():
    source = BasketStateSource(
        basket_id="ETH-USDT-SWAP-momentum_burst",
        symbol="ETH-USDT-SWAP",
        strategy="momentum_burst",
        position=OkxPositionSource("ETH-USDT-SWAP", "buy", 2.0, 3000.0),
        orders=(
            OkxOrderSource(
                "ETH-USDT-SWAP",
                "buy",
                2.0,
                3001.0,
                "o1",
                "c1",
                "ETH-USDT-SWAP-momentum_burst",
            ),
        ),
    )
    evaluated = evaluate_basket_source(source)
    assert isinstance(evaluated, EvaluatedBasketState)
    assert evaluated.state == BasketState.ACTIVE
    assert evaluated.current_sz == 2.0


def test_basket_and_okx_client_id_helpers_encode_state_identity():
    basket_id = format_basket_id("ETH-USDT-SWAP", "momentum_burst", "reversal")
    assert parse_basket_id(basket_id) == ("ETH-USDT-SWAP", "momentum_burst", "reversal")
    client_id = format_okx_client_id(basket_id, "enter")
    assert client_id.startswith("qooi")
    assert len(client_id) <= 32
    assert client_id.isalnum()
