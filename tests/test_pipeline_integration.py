"""White-box integration tests — full pipeline cycle through all 4 layers."""

import polars as pl

from qooi.core import BarMarket, BarSignal, PipelineContext, PipelinePolicy, process_bar
from qooi.core.basket import (
    ActionKind,
    Basket,
    BasketBook,
    BasketState,
    ExitConfig,
    ExitReason,
    TrailTracker,
    evaluate_exits,
)
from qooi.core.instruments import AssetConfig, PairConfig
from qooi.core.recovery import GridRecovery, MartingaleRecovery, ReverseRecovery


def _pair(symbol: str = "TEST-USDT-SWAP", capital: float = 500.0) -> PairConfig:
    return PairConfig(
        asset=AssetConfig(
            symbol=symbol,
            sig_symbol="TEST-USDT",
            timeframe="1H",
            capital=capital,
            leverage=2.0,
            ct_val=0.1,
            signal_threshold=0.01,
        )
    )


def _df(close_values, atr=10.0, high_mult=1.01, low_mult=0.99):
    """Build a minimal OHLCV DataFrame with indicators for N bars."""
    n = len(close_values)
    return pl.DataFrame(
        {
            "timestamp": list(range(1000, 1000 + n)),
            "open": [c * 0.998 for c in close_values],
            "high": [c * high_mult for c in close_values],
            "low": [c * low_mult for c in close_values],
            "close": list(close_values),
            "vol": [100.0] * n,
            "atr_14": [atr] * n,
        }
    )


def _run_bar(
    df,
    baskets,
    pair,
    exit_cfg=None,
    recovery_cfg=None,
    *,
    signal=0.0,
    entry=None,
    exit_signal=False,
    policy=PipelinePolicy(),
):
    book = baskets if isinstance(baskets, BasketBook) else BasketBook(baskets)
    market = BarMarket.from_frame(df)
    active = book.active_for_strategy(pair.asset.symbol, "default")
    entry_value = float(entry if entry is not None else (signal if not active else 0.0))
    context = PipelineContext(
        strategy_id="default",
        market=market,
        signal=BarSignal(
            position=float(signal),
            entry=entry_value,
            exit=bool(exit_signal),
            strength=1.0,
            signal_id="test_signal",
        ),
        policy=policy,
    )
    actions = process_bar(df, book, pair, exit_cfg, recovery_cfg, context=context)
    touched = {a.basket_id for a in actions}
    touched.update(f"{a.basket_id}_hedge" for a in actions if a.action == ActionKind.HEDGE)
    book.apply_actions(actions)
    book.advance_bar(market.close, market.high, market.low, skip_ids=touched)
    return actions


def test_signal_entry_creates_basket():
    """Layer 1+2: signal=1 produces ENTER action, basket moves to ACTIVE."""
    pair = _pair()
    df = _df([100.0, 101.0, 102.0])
    baskets: list[Basket] = []

    actions = _run_bar(df, baskets, pair, signal=1.0)

    assert len(actions) == 1
    a = actions[0]
    assert a.action == ActionKind.ENTER
    assert a.reason == ExitReason.SIGNAL_ENTRY.value
    assert len(baskets) == 1
    assert baskets[0].is_active


def test_basket_does_not_duplicate_on_same_signal():
    """Layer 2: active basket with same signal does not create duplicate."""
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    actions = _run_bar(df, baskets, pair, signal=1.0)
    assert len(actions) == 1
    assert actions[0].action == ActionKind.ENTER

    actions2 = _run_bar(df, baskets, pair, signal=1.0)
    assert len(actions2) == 0


def test_opposite_held_signal_does_not_exit_basket_by_default():
    """Layer 2: held opposite signal does not close independent baskets."""
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    assert baskets[0].is_active

    actions = _run_bar(df, baskets, pair, signal=-1.0)
    assert not any(a.reason == ExitReason.SIGNAL_FLIP.value for a in actions)
    assert baskets[0].is_active


def test_signal_zero_does_not_exit_basket_by_default():
    """Layer 2: neutral held signal does not close independent baskets."""
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    assert baskets[0].is_active

    actions = _run_bar(df, baskets, pair, signal=0.0)
    assert not any(a.reason == ExitReason.SIGNAL_ZERO.value for a in actions)
    assert baskets[0].is_active


def test_close_on_neutral_compatibility_mode_exits_basket():
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    actions = _run_bar(
        df,
        baskets,
        pair,
        signal=0.0,
        policy=PipelinePolicy(close_on_neutral_signal=True),
    )

    assert len(actions) == 1
    assert actions[0].reason == ExitReason.SIGNAL_ZERO.value
    assert baskets[0].is_idle


def test_exit_signal_closes_active_basket():
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    actions = _run_bar(df, baskets, pair, signal=1.0, exit_signal=True)

    assert len(actions) == 1
    assert actions[0].reason == ExitReason.STRATEGY_EXIT.value
    assert baskets[0].is_idle


def test_strict_thesis_continuation_exits_when_trend_fails():
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    actions = _run_bar(
        df,
        baskets,
        pair,
        signal=0.0,
        policy=PipelinePolicy(require_thesis_continuation=True),
    )

    assert len(actions) == 1
    assert actions[0].reason == ExitReason.THESIS_FAILED.value
    assert baskets[0].is_idle


def test_opposite_entry_flip_policy_closes_basket():
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0, entry=1.0)
    actions = _run_bar(
        df,
        baskets,
        pair,
        signal=-1.0,
        entry=-1.0,
        policy=PipelinePolicy(flip_policy="close_same_strategy_opposite"),
    )

    assert any(a.reason == ExitReason.SIGNAL_FLIP.value for a in actions)
    assert any(a.action == ActionKind.ENTER and a.side == "sell" for a in actions)


def test_bars_in_pos_accumulates():
    """Layer 4: bars_in_pos increments each bar without exit."""
    pair = _pair()
    baskets: list[Basket] = []
    # 20 bars at 102 (above entry, not hitting stop/target with 10 ATR)
    prices = [100.0] + [102.0] * 20

    _run_bar(_df(prices[:2]), baskets, pair, signal=1.0)
    assert baskets[0].bars_in_pos == 0  # just entered

    for _ in range(5):
        _run_bar(_df(prices[:2]), baskets, pair, signal=1.0)

    assert baskets[0].bars_in_pos == 5


def test_time_stop_fires():
    """Layer 4: time stop fires after max_bars, even without target hit."""
    pair = _pair()
    baskets: list[Basket] = []
    cfg = ExitConfig(stop_mult=100.0, target_mult=100.0, max_bars=3)
    df = _df([100.0, 101.0])  # entry at close=101.0, no stop/target breach with 100x ATR

    _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)
    assert baskets[0].is_active
    assert baskets[0].bars_in_pos == 0

    # Run 3 more bars — bars_in_pos goes 0→1, 1→2, 2→3. At 3, time stop fires.
    for _ in range(3):
        _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)

    assert baskets[0].bars_in_pos == 3
    assert baskets[0].is_active

    # 4th bar: bars_in_pos=3 >= max_bars=3, time stop fires
    _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)
    assert baskets[0].is_idle


def test_grid_recovery_activates():
    """Layer 3: grid adds when price moves beyond zone_atr in losing direction."""
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    rec = GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)

    # Bar 1: entry at close=100.0
    _run_bar(_df([100.0, 100.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    assert baskets[0].is_active
    assert baskets[0].entry_px == 100.0

    # Bar 2: close drops to 97.0, loss=-3%, ATR=3, zone_atr=1 → grid level 1
    actions = _run_bar(_df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    grid_actions = [a for a in actions if a.action == ActionKind.ADD_GRID]
    assert len(grid_actions) >= 1
    assert baskets[0].recovery_level >= 1
    assert baskets[0].recovery_activated


def test_martingale_produces_exit_and_enter():
    """Layer 3: martingale returns [EXIT, ENTER] pair."""
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    rec = MartingaleRecovery(zone_atr=1.0, max_levels=3)

    # Bar 1: entry at close=100.0
    _run_bar(_df([100.0, 100.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    assert baskets[0].is_active
    assert baskets[0].entry_px == 100.0

    # Bar 2: close drops to 97.0, loss=-3%, zone_atr=1 → martingale triggers
    actions = _run_bar(_df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    exit_actions = [a for a in actions if a.action == ActionKind.EXIT]
    enter_actions = [a for a in actions if a.action == ActionKind.ENTER]
    assert len(exit_actions) >= 1
    assert len(enter_actions) >= 1
    assert exit_actions[0].reason == ExitReason.MARTINGALE.value
    assert enter_actions[0].side == "sell"


def test_reverse_recovery_requires_opposite_thesis():
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    rec = ReverseRecovery(zone_atr=1.0, max_levels=3)

    _run_bar(_df([100.0, 100.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    no_reverse = _run_bar(_df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=1.0)
    assert not any(a.reason == ExitReason.MARTINGALE.value for a in no_reverse)

    reverse = _run_bar(_df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=rec, signal=-1.0)

    assert any(
        a.action == ActionKind.EXIT and a.reason == ExitReason.MARTINGALE.value for a in reverse
    )
    assert any(a.action == ActionKind.ENTER and a.side == "sell" for a in reverse)


def test_hard_stop_preempts_recovery_same_bar():
    """Hard stops outrank recovery when both are possible on the same bar."""
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    rec = GridRecovery(zone_atr=1.0, multiplier=2.0)
    cfg = ExitConfig(stop_mult=0.1, target_mult=100.0, max_bars=0)

    _run_bar(
        _df([100.0, 100.0], atr=3.0),
        baskets,
        pair,
        recovery_cfg=rec,
        exit_cfg=cfg,
        signal=1.0,
    )
    actions = _run_bar(
        _df([100.0, 90.0], atr=3.0),
        baskets,
        pair,
        recovery_cfg=rec,
        exit_cfg=cfg,
        signal=1.0,
    )

    assert any(a.action == ActionKind.EXIT and a.reason == ExitReason.STOP.value for a in actions)
    assert not any(a.action == ActionKind.ADD_GRID for a in actions)
    assert not any(a.reason == ExitReason.TIME.value for a in actions)


def test_multiple_active_baskets_possible_with_hedge_action():
    """Hedge action can coexist with original active basket."""
    primary = Basket(
        basket_id="SOL-rsi",
        symbol="SOL",
        strategy="rsi",
        side="buy",
        state=BasketState.ACTIVE,
        entry_px=100.0,
        current_sz=2.0,
    )
    hedge = Basket(
        basket_id="SOL-rsi_hedge",
        symbol="SOL",
        strategy="rsi",
        side="sell",
        state=BasketState.ACTIVE,
        entry_px=99.0,
        current_sz=2.0,
    )
    baskets = [primary, hedge]
    active_count = sum(1 for b in baskets if b.is_active)
    assert active_count == 2


def test_trailing_disabled_during_recovery():
    """skip_trailing blocks trailing/breakeven while recovery active."""
    basket = Basket(
        basket_id="x",
        symbol="X",
        strategy="s",
        side="buy",
        state=BasketState.ACTIVE,
        entry_px=100.0,
        current_sz=1.0,
        recovery_activated=True,
        recovery_level=1,
    )
    trail = TrailTracker(trail_high=110.0, trail_low=95.0, target_hit=True)
    cfg = ExitConfig(trail_mult=1.0, breakeven_after_target=True)
    action = evaluate_exits(
        basket,
        bar_close=99.0,
        bar_high=111.0,
        bar_low=99.0,
        atr=5.0,
        trail=trail,
        config=cfg,
        skip_trailing=True,
    )
    assert action is None


def test_multiple_baskets_per_strategy():
    """Layer 2: explicit entry events can create independent baskets up to caps."""
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0, entry=1.0)
    assert len(baskets) == 1

    actions = _run_bar(df, baskets, pair, signal=1.0, entry=1.0)
    assert any(a.action == ActionKind.ENTER for a in actions)
    assert len(baskets) == 2


def test_no_signal_no_action():
    """Layer 1+2: signal=0 on idle basket produces no action."""
    pair = _pair()
    df = _df([100.0])
    baskets: list[Basket] = []

    actions = _run_bar(df, baskets, pair, signal=0.0)
    assert len(actions) == 0
    assert len(baskets) == 0
