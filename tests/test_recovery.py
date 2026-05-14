"""Unit tests for recovery layer."""

from qooi.core.basket import ActionKind, Basket, BasketState, ExitReason
from qooi.core.recovery import RecoveryConfig, RecoveryKind, evaluate


def _b(side="buy", entry_px=100.0, sz=10.0, level=0):
    return Basket(
        basket_id="test",
        symbol="X",
        strategy="s",
        side=side,
        entry_px=entry_px,
        current_sz=sz,
        recovery_level=level,
        state=BasketState.ACTIVE,
    )


def test_grid_add_on_drawdown():
    b = _b("buy", 100.0, sz=1.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.GRID, zone_atr=1.0, multiplier=2.0, max_levels=3)
    a = evaluate(b, bar_close=97.0, atr=2.0, config=cfg, current_level=0)
    assert a is not None
    assert a.action == ActionKind.ADD_GRID
    assert a.side == "buy"
    assert a.sz == 2.0  # 1.0 * 2.0


def test_grid_not_triggered_within_zone():
    b = _b("buy", 100.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.GRID, zone_atr=1.0, multiplier=2.0, max_levels=3)
    a = evaluate(b, bar_close=99.0, atr=2.0, config=cfg, current_level=0)
    assert a is None


def test_grid_max_levels():
    b = _b("buy", 100.0, sz=1.0, level=3)
    cfg = RecoveryConfig(strategy=RecoveryKind.GRID, zone_atr=1.0, multiplier=2.0, max_levels=3)
    a = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=3)
    assert a is None


def test_grid_guard_zero_sz():
    b = _b("buy", 100.0, sz=0.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.GRID, zone_atr=1.0, multiplier=2.0)
    a = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert a is None


def test_martingale_reverse():
    b = _b("buy", 100.0, sz=1.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.MARTINGALE, zone_atr=2.0, max_levels=3)
    a = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0)
    assert a is not None
    assert a.action == ActionKind.EXIT
    assert a.reason == ExitReason.MARTINGALE.value


def test_hedge_on_drawdown():
    b = _b("buy", 100.0, sz=1.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.HEDGE, zone_atr=2.0)
    a = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0)
    assert a is not None
    assert a.action == ActionKind.HEDGE
    assert a.side == "sell"


def test_none_strategy_no_action():
    b = _b("buy", 100.0)
    cfg = RecoveryConfig(strategy=RecoveryKind.NONE)
    a = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert a is None


def test_idle_basket_skipped():
    b = _b("buy", 100.0)
    b.state = BasketState.IDLE
    cfg = RecoveryConfig(strategy=RecoveryKind.GRID, zone_atr=1.0)
    a = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert a is None
