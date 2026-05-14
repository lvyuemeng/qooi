"""Unit tests for basket management layer."""

from qooi.core.basket import ActionKind, Basket, BasketAction, BasketManager, BasketState, Position


def test_create_basket():
    mgr = BasketManager(max_baskets=3, max_per_symbol=1)
    b = mgr.create("ETH-USDT-SWAP", "momentum_1h", "buy", 3000.0)
    assert b.basket_id == "ETH-USDT-SWAP-momentum_1h"
    assert b.state == BasketState.ACTIVE
    assert b.side == "buy"
    assert b.entry_px == 3000.0
    assert b.current_sz == 0.0


def test_manager_dedup_per_symbol():
    mgr = BasketManager(max_baskets=5, max_per_symbol=1)
    active: list[Basket] = []
    b1 = mgr.create("ETH-USDT-SWAP", "m1", "buy", 100.0)
    active.append(b1)
    assert mgr.can_open("ETH-USDT-SWAP", active) is False
    assert mgr.can_open("SOL-USDT-SWAP", active) is True


def test_cannot_open_when_full():
    mgr = BasketManager(max_baskets=2, max_per_symbol=5)
    active = [
        Basket(basket_id="a", symbol="X", strategy="s", side="buy"),
        Basket(basket_id="b", symbol="Y", strategy="s", side="sell"),
    ]
    assert mgr.can_open("Z", active) is False


def test_basket_properties():
    b = Basket(basket_id="test", symbol="ETH", strategy="m", side="buy", state=BasketState.IDLE)
    assert b.is_idle is True
    assert b.is_active is False
    b.state = BasketState.ACTIVE
    assert b.is_active is True


def test_basket_action_defaults():
    a = BasketAction(basket_id="x", action=ActionKind.ENTER, reason="signal")
    assert a.side == ""
    assert a.px == 0.0
    assert a.fraction == 1.0
    assert a.order_type == "limit"


def test_position_defaults():
    p = Position(symbol="ETH", side="buy", sz=1.0, avg_px=2000.0)
    assert p.order_id == ""


def test_manager_remove():
    mgr = BasketManager()
    b = Basket(
        basket_id="x",
        symbol="A",
        strategy="s",
        side="buy",
        state=BasketState.ACTIVE,
        positions=[Position("A", "buy", 1.0, 100.0)],
    )
    mgr.remove(b)
    assert b.state == BasketState.IDLE
    assert len(b.positions) == 0
