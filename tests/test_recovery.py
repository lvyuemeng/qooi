"""Unit tests for recovery layer."""

from qooi.core.basket import ActionKind, Basket, BasketState, ExitReason
from qooi.core.recovery import (
    GridRecovery,
    HedgeRecovery,
    MartingaleRecovery,
    NoRecovery,
    ZScoreReversionRecovery,
    evaluate,
)


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


def _first(actions):
    assert len(actions) > 0
    return actions[0]


def test_grid_add_on_drawdown():
    b = _b("buy", 100.0, sz=1.0)
    cfg = GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)
    actions = evaluate(b, bar_close=97.0, atr=2.0, config=cfg, current_level=0)
    a = _first(actions)
    assert a.action == ActionKind.ADD_GRID
    assert a.side == "buy"
    assert a.sz == 2.0


def test_grid_not_triggered_within_zone():
    b = _b("buy", 100.0)
    cfg = GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)
    actions = evaluate(b, bar_close=99.0, atr=2.0, config=cfg, current_level=0)
    assert len(actions) == 0


def test_grid_max_levels():
    b = _b("buy", 100.0, sz=1.0, level=3)
    cfg = GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)
    actions = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=3)
    assert len(actions) == 0


def test_grid_guard_zero_sz():
    b = _b("buy", 100.0, sz=0.0)
    cfg = GridRecovery(zone_atr=1.0, multiplier=2.0)
    actions = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert len(actions) == 0


def test_martingale_reverse():
    b = _b("buy", 100.0, sz=1.0)
    cfg = MartingaleRecovery(zone_atr=2.0, max_levels=3)
    actions = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0)
    assert len(actions) == 2
    assert actions[0].action == ActionKind.EXIT
    assert actions[0].reason == ExitReason.MARTINGALE.value
    assert actions[1].action == ActionKind.ENTER
    assert actions[1].side == "sell"


def test_hedge_on_drawdown():
    b = _b("buy", 100.0, sz=1.0)
    cfg = HedgeRecovery(zone_atr=2.0)
    actions = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0)
    a = _first(actions)
    assert a.action == ActionKind.HEDGE
    assert a.side == "sell"


def test_none_strategy_no_action():
    b = _b("buy", 100.0)
    cfg = NoRecovery()
    actions = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert len(actions) == 0


def test_max_loss_exits_before_recovery_add():
    b = _b("buy", 100.0, sz=1.0)
    cfg = GridRecovery(zone_atr=1.0, max_loss_pct=2.0)
    actions = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0)
    a = _first(actions)
    assert a.action == ActionKind.EXIT
    assert a.reason == ExitReason.GLOBAL_LOSS_LIMIT.value


def test_martingale_size_uses_contract_value_units():
    b = _b("buy", 100.0, sz=10.0)
    cfg = MartingaleRecovery(zone_atr=1.0, max_levels=3)
    actions = evaluate(b, bar_close=95.0, atr=2.0, config=cfg, current_level=0, ct_val=0.1)
    assert actions[1].sz == 25


def test_idle_basket_skipped():
    b = _b("buy", 100.0)
    b.state = BasketState.IDLE
    cfg = GridRecovery(zone_atr=1.0)
    actions = evaluate(b, bar_close=0.0, atr=2.0, config=cfg, current_level=0)
    assert len(actions) == 0


def test_zscore_recovery_adds_on_adverse_move_with_reversion_evidence():
    b = _b("buy", 100.0, sz=1.0)
    cfg = ZScoreReversionRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1)
    actions = evaluate(
        b,
        bar_close=97.0,
        atr=2.0,
        config=cfg,
        current_level=0,
        zscore=-2.2,
        zscore_delta=0.2,
        short_momentum_return=0.01,
        lower_wick_ratio=0.2,
        volatility_ratio=1.0,
        trend_return=0.0,
        adx=20.0,
    )
    a = _first(actions)
    assert a.action == ActionKind.ADD_GRID
    assert a.reason == "zscore_recovery_level_1"
    assert a.sz == 1.0


def test_zscore_recovery_blocks_expanding_zscore():
    b = _b("buy", 100.0, sz=1.0)
    cfg = ZScoreReversionRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1)
    actions = evaluate(
        b,
        bar_close=97.0,
        atr=2.0,
        config=cfg,
        current_level=0,
        zscore=-2.2,
        zscore_delta=-0.2,
        short_momentum_return=0.01,
        lower_wick_ratio=0.5,
        volatility_ratio=1.0,
        trend_return=0.0,
        adx=20.0,
    )
    assert actions == []


def test_zscore_recovery_blocks_volatility_expansion():
    b = _b("sell", 100.0, sz=1.0)
    cfg = ZScoreReversionRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1)
    actions = evaluate(
        b,
        bar_close=103.0,
        atr=2.0,
        config=cfg,
        current_level=0,
        zscore=2.2,
        zscore_delta=-0.2,
        short_momentum_return=-0.01,
        upper_wick_ratio=0.5,
        volatility_ratio=2.0,
        trend_return=0.0,
        adx=20.0,
    )
    assert actions == []


