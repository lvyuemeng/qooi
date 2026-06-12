"""Basket invariants."""

from qooi.core.basket import (
    ActionKind,
    Basket,
    BasketAction,
    BasketBook,
    BasketManager,
    BasketState,
    Position,
)
from qooi.core.config import AssetConfig


def test_create_basket_initializes_full_state():
    mgr = BasketManager(max_baskets=3, max_per_symbol=1)
    b = mgr.create(
        "ETH-USDT-SWAP",
        "momentum_burst",
        "buy",
        3000.0,
        sz=2.0,
        stop_px=2950.0,
        target_px=3060.0,
    )
    assert b.basket_id == "ETH-USDT-SWAP-momentum_burst-long-1-1"
    assert b.state == BasketState.ACTIVE
    assert b.entry_px == 3000.0
    assert b.current_sz == 2.0
    assert b.stop_px == 2950.0
    assert b.target_px == 3060.0


def test_can_open_counts_only_active_baskets():
    mgr = BasketManager(max_baskets=5, max_per_symbol=1)
    baskets = [
        Basket(
            basket_id="SOL-rsi",
            symbol="SOL",
            strategy="rsi",
            side="buy",
            state=BasketState.ACTIVE,
        )
    ]
    assert mgr.can_open("SOL", baskets) is False
    mgr.remove(baskets[0])
    assert mgr.can_open("SOL", baskets) is True


def test_max_basket_limit_blocks_new_entries():
    mgr = BasketManager(max_baskets=2, max_per_symbol=5)
    baskets = [
        Basket(basket_id="a", symbol="X", strategy="s", side="buy", state=BasketState.ACTIVE),
        Basket(basket_id="b", symbol="Y", strategy="s", side="sell", state=BasketState.ACTIVE),
    ]
    assert mgr.can_open("Z", baskets) is False


def test_remove_fully_resets_basket():
    mgr = BasketManager()
    b = Basket(
        basket_id="x",
        symbol="A",
        strategy="s",
        side="buy",
        state=BasketState.ACTIVE,
        positions=[Position("A", "buy", 1.0, 100.0)],
        bars_in_pos=5,
        trail_high=110.0,
        target_hit=True,
        recovery_activated=True,
        recovery_level=2,
        cumulative_loss=42.0,
        current_sz=3.0,
    )
    mgr.remove(b)
    assert b.state == BasketState.IDLE
    assert b.positions == []
    assert b.bars_in_pos == 0
    assert b.trail_high == 0.0
    assert b.trail_low == float("inf")
    assert b.target_hit is False
    assert b.recovery_activated is False
    assert b.recovery_level == 0
    assert b.cumulative_loss == 0.0
    assert b.current_sz == 0.0


def test_replace_invariant_same_symbol_reopens_without_list_growth():
    mgr = BasketManager(max_per_symbol=1)
    baskets = [
        Basket(
            basket_id="SOL-rsi",
            symbol="SOL",
            strategy="rsi",
            side="buy",
            state=BasketState.ACTIVE,
        )
    ]
    mgr.remove(baskets[0])
    assert mgr.can_open("SOL", baskets) is True


def test_basket_book_owns_lifecycle_state():
    book = BasketBook(max_per_symbol=1)
    basket = book.open("ETH", "momentum", "buy", 100.0, 2.0, 95.0, 110.0)

    assert book.get(basket.basket_id) is basket
    assert book.active_exposure() == 2.0
    assert book.can_open("ETH") is False

    book.close(basket)
    assert basket.is_idle
    assert book.can_open("ETH") is True


def test_apply_action_owns_recovery_state_mutation():
    book = BasketBook()
    basket = book.open("ETH", "momentum", "buy", 100.0, 2.0, 95.0, 110.0)

    book.apply_action(BasketAction(basket.basket_id, ActionKind.ADD_GRID, sz=1.0, px=98.0))

    assert basket.recovery_level == 1
    assert basket.recovery_activated is True


def test_size_decision_reports_binding_cap_and_can_block_min_contract():
    cfg = AssetConfig(
        symbol="ETH",
        capital=100.0,
        leverage=1.0,
        ct_val=1.0,
        max_risk_pct=0.01,
        max_notional_pct_per_basket=0.01,
        min_contracts=1,
    )

    decision = BasketManager.size_decision(100.0, 95.0, cfg)

    assert decision.contracts == 0
    assert decision.binding_cap == "risk"
    assert decision.blocked_reason == "below_min_contracts_1"


def test_size_decision_supports_fractional_exchange_lots():
    cfg = AssetConfig(
        symbol="BTC",
        capital=500.0,
        leverage=2.0,
        ct_val=0.01,
        max_risk_pct=0.02,
        max_notional_pct_per_basket=1.0,
        min_contracts=0.01,
        lot_size=0.01,
    )

    decision = BasketManager.size_decision(100000.0, 99500.0, cfg, signal_strength=0.25)

    assert decision.contracts == 0.5
    assert decision.blocked_reason == ""
    assert decision.risk_sized_contracts == 0.5


def test_stop_target_rounds_to_configured_tick_size():
    cfg = AssetConfig(symbol="BTC", ct_val=0.01, tick_size=0.1)

    stop_px, target_px = BasketManager.compute_stop_target("buy", 100.03, 1.04, cfg)

    assert stop_px == 98.0
    assert target_px == 103.2
