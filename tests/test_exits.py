"""Unit tests for exit engine."""

from qooi.core.basket import (
    ActionKind,
    Basket,
    ExitConfig,
    ExitReason,
    TrailTracker,
    evaluate_exits,
    evaluate_hard_exits,
)


def _b(side="buy", entry_px=100.0, bars=0):
    return Basket(
        basket_id="test",
        symbol="X",
        strategy="s",
        side=side,
        entry_px=entry_px,
        bars_in_pos=bars,
        current_sz=1.0,
    )


def test_stop_hit_long():
    b = _b("buy", 100.0)
    t = TrailTracker()
    cfg = ExitConfig(stop_mult=1.0, target_mult=10.0, max_bars=100)
    # atr=5, stop = 100 - 1*5 = 95
    a = evaluate_exits(b, bar_close=94.0, bar_high=100, bar_low=94, atr=5.0, trail=t, config=cfg)
    assert a is not None
    assert a.action == ActionKind.EXIT
    assert a.reason == ExitReason.STOP.value


def test_hard_exit_stop_matches_normal_stop_for_long_and_short():
    cfg = ExitConfig(stop_mult=1.0, target_mult=10.0, max_bars=100)
    cases = [
        (_b("buy", 100.0), 94.0, 100.0, 94.0),
        (_b("sell", 100.0), 106.0, 106.0, 100.0),
    ]

    for basket, close, high, low in cases:
        normal = evaluate_exits(
            basket,
            bar_close=close,
            bar_high=high,
            bar_low=low,
            atr=5.0,
            trail=TrailTracker(),
            config=cfg,
        )
        hard = evaluate_hard_exits(
            basket,
            bar_close=close,
            bar_high=high,
            bar_low=low,
            atr=5.0,
            config=cfg,
        )
        assert normal is not None
        assert hard is not None
        assert normal.reason == hard.reason == ExitReason.STOP.value
        assert normal.px == hard.px


def test_hard_exit_ignores_stop_after_target_already_active():
    basket = _b("buy", 100.0)
    basket.target_hit = True
    cfg = ExitConfig(stop_mult=1.0, target_mult=1.0, max_bars=100)

    hard = evaluate_hard_exits(
        basket,
        bar_close=94.0,
        bar_high=110.0,
        bar_low=94.0,
        atr=5.0,
        config=cfg,
    )

    assert hard is None


def test_stop_not_hit_if_target_first():
    b = _b("buy", 100.0)
    t = TrailTracker(target_hit=True)  # target already hit, trailing activated
    cfg = ExitConfig(stop_mult=1.0, target_mult=1.0, trail_mult=2.0, max_bars=100)
    a = evaluate_exits(b, bar_close=94.0, bar_high=110, bar_low=94, atr=5.0, trail=t, config=cfg)
    assert a is None or a.reason != ExitReason.STOP.value


def test_target_activates_trail():
    b = _b("buy", 100.0)
    t = TrailTracker()
    cfg = ExitConfig(target_mult=1.0, trail_mult=2.0, max_bars=100)
    a = evaluate_exits(b, bar_close=106.0, bar_high=106, bar_low=105, atr=5.0, trail=t, config=cfg)
    assert t.target_hit is True
    assert a is None


def test_target_activation_does_not_trail_exit_until_next_bar():
    b = _b("buy", 100.0)
    t = TrailTracker()
    cfg = ExitConfig(stop_mult=10.0, target_mult=1.0, trail_mult=0.5, max_bars=100)

    a = evaluate_exits(
        b,
        bar_close=101.0,
        bar_high=106.0,
        bar_low=100.0,
        atr=5.0,
        trail=t,
        config=cfg,
    )

    assert t.target_hit is True
    assert a is None


def test_trailing_stop_after_target():
    b = _b("buy", 100.0)
    t = TrailTracker(target_hit=True, trail_high=106.0)
    cfg = ExitConfig(trail_mult=2.0, stop_mult=10.0, max_bars=100)
    # trail_stop = 106 - 2*5 = 96
    a = evaluate_exits(b, bar_close=95.0, bar_high=100, bar_low=95, atr=5.0, trail=t, config=cfg)
    assert a is not None
    assert a.reason == ExitReason.TRAILING.value


def test_skip_trailing_blocks_trailing_and_breakeven_during_recovery():
    b = _b("buy", 100.0)
    b.recovery_activated = True
    b.recovery_level = 1
    t = TrailTracker(trail_high=110.0, trail_low=95.0, target_hit=True)
    cfg = ExitConfig(trail_mult=1.0, breakeven_after_target=True)

    a = evaluate_exits(
        b,
        bar_close=99.0,
        bar_high=111.0,
        bar_low=99.0,
        atr=5.0,
        trail=t,
        config=cfg,
        skip_trailing=True,
    )

    assert a is None


def test_time_stop():
    b = _b("buy", 100.0, bars=11)
    t = TrailTracker()
    cfg = ExitConfig(max_bars=10, target_mult=100.0)
    a = evaluate_exits(b, bar_close=101.0, bar_high=101, bar_low=101, atr=5.0, trail=t, config=cfg)
    assert a is not None
    assert a.reason == ExitReason.TIME.value


def test_breakeven_stop():
    b = _b("buy", 100.0)
    t = TrailTracker(target_hit=True)
    cfg = ExitConfig(breakeven_after_target=True, stop_mult=10.0, target_mult=10.0)
    a = evaluate_exits(b, bar_close=99.0, bar_high=99, bar_low=99, atr=5.0, trail=t, config=cfg)
    assert a is not None
    assert a.reason == ExitReason.BREAKEVEN.value


def test_no_exit_when_in_range():
    b = _b("buy", 100.0)
    t = TrailTracker()
    cfg = ExitConfig(stop_mult=10.0, target_mult=10.0, max_bars=100)
    a = evaluate_exits(b, bar_close=101.0, bar_high=101, bar_low=100, atr=5.0, trail=t, config=cfg)
    assert a is None


def test_trail_updates_high_low():
    t = TrailTracker()
    t.update(105.0, 95.0)
    assert t.trail_high == 105.0
    assert t.trail_low == 95.0
    t.update(104.0, 96.0)
    assert t.trail_high == 105.0
    assert t.trail_low == 95.0
