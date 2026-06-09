"""Core process-bar boundary tests."""

import polars as pl
import pytest

from qooi.core import BarMarket, BarSignal, PipelineContext, PipelinePolicy, process_bar
from qooi.core.basket import ActionKind, Basket, BasketBook, ExitConfig, ExitReason
from qooi.core.config import AssetConfig, PairConfig
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


def test_idle_and_duplicate_signal_entry_contracts():
    pair = _pair()
    df = _df([100.0, 101.0, 102.0])
    baskets: list[Basket] = []

    assert _run_bar(df, baskets, pair, signal=0.0) == []
    actions = _run_bar(df, baskets, pair, signal=1.0)
    duplicate = _run_bar(df, baskets, pair, signal=1.0)

    assert len(actions) == 1
    assert actions[0].action == ActionKind.ENTER
    assert actions[0].reason == ExitReason.SIGNAL_ENTRY.value
    assert baskets[0].is_active
    assert duplicate == []


@pytest.mark.parametrize(
    "held_signal, unexpected_reason",
    [
        (-1.0, ExitReason.SIGNAL_FLIP.value),
        (0.0, ExitReason.SIGNAL_ZERO.value),
    ],
)
def test_held_signal_changes_do_not_close_independent_baskets_by_default(
    held_signal,
    unexpected_reason,
):
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    actions = _run_bar(df, baskets, pair, signal=held_signal)

    assert not any(a.reason == unexpected_reason for a in actions)
    assert baskets[0].is_active


@pytest.mark.parametrize(
    "policy, signal, exit_signal, reason",
    [
        (PipelinePolicy(close_on_neutral_signal=True), 0.0, False, ExitReason.SIGNAL_ZERO.value),
        (
            PipelinePolicy(require_thesis_continuation=True),
            0.0,
            False,
            ExitReason.THESIS_FAILED.value,
        ),
        (PipelinePolicy(), 1.0, True, ExitReason.STRATEGY_EXIT.value),
    ],
)
def test_policy_exit_modes_close_active_baskets(policy, signal, exit_signal, reason):
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0)
    actions = _run_bar(df, baskets, pair, signal=signal, exit_signal=exit_signal, policy=policy)

    assert len(actions) == 1
    assert actions[0].reason == reason
    assert baskets[0].is_idle


def test_opposite_entry_flip_policy_closes_and_reverses_basket():
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


def test_bar_lifecycle_tracks_holding_period_and_time_stop():
    pair = _pair()
    baskets: list[Basket] = []
    cfg = ExitConfig(stop_mult=100.0, target_mult=100.0, max_bars=3)
    df = _df([100.0, 101.0])

    _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)
    assert baskets[0].bars_in_pos == 0

    for expected in range(1, 4):
        _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)
        assert baskets[0].bars_in_pos == expected
        assert baskets[0].is_active

    _run_bar(df, baskets, pair, exit_cfg=cfg, signal=1.0)
    assert baskets[0].is_idle


def test_explicit_entry_events_can_create_independent_baskets():
    pair = _pair()
    df = _df([100.0, 101.0])
    baskets: list[Basket] = []

    _run_bar(df, baskets, pair, signal=1.0, entry=1.0)
    actions = _run_bar(df, baskets, pair, signal=1.0, entry=1.0)

    assert any(a.action == ActionKind.ENTER for a in actions)
    assert len(baskets) == 2


def test_grid_and_martingale_recovery_actions_flow_through_process_bar():
    pair = _pair(capital=500.0)

    grid_baskets: list[Basket] = []
    _run_bar(
        _df([100.0, 100.0], atr=3.0),
        grid_baskets,
        pair,
        recovery_cfg=GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3),
        signal=1.0,
    )
    grid_actions = _run_bar(
        _df([100.0, 97.0], atr=3.0),
        grid_baskets,
        pair,
        recovery_cfg=GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3),
        signal=1.0,
    )

    martingale_baskets: list[Basket] = []
    _run_bar(
        _df([100.0, 100.0], atr=3.0),
        martingale_baskets,
        pair,
        recovery_cfg=MartingaleRecovery(zone_atr=1.0, max_levels=3),
        signal=1.0,
    )
    martingale_actions = _run_bar(
        _df([100.0, 97.0], atr=3.0),
        martingale_baskets,
        pair,
        recovery_cfg=MartingaleRecovery(zone_atr=1.0, max_levels=3),
        signal=1.0,
    )

    assert any(a.action == ActionKind.ADD_GRID for a in grid_actions)
    assert grid_baskets[0].recovery_level >= 1
    assert grid_baskets[0].recovery_activated
    assert any(
        a.action == ActionKind.EXIT and a.reason == ExitReason.MARTINGALE.value
        for a in martingale_actions
    )
    assert any(a.action == ActionKind.ENTER and a.side == "sell" for a in martingale_actions)


def test_reverse_recovery_requires_opposite_thesis():
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    recovery = ReverseRecovery(zone_atr=1.0, max_levels=3)

    _run_bar(_df([100.0, 100.0], atr=3.0), baskets, pair, recovery_cfg=recovery, signal=1.0)
    no_reverse = _run_bar(
        _df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=recovery, signal=1.0
    )
    reverse = _run_bar(
        _df([100.0, 97.0], atr=3.0), baskets, pair, recovery_cfg=recovery, signal=-1.0
    )

    assert not any(a.reason == ExitReason.MARTINGALE.value for a in no_reverse)
    assert any(
        a.action == ActionKind.EXIT and a.reason == ExitReason.MARTINGALE.value for a in reverse
    )
    assert any(a.action == ActionKind.ENTER and a.side == "sell" for a in reverse)


def test_hard_stop_preempts_recovery_same_bar():
    pair = _pair(capital=500.0)
    baskets: list[Basket] = []
    recovery = GridRecovery(zone_atr=1.0, multiplier=2.0)
    cfg = ExitConfig(stop_mult=0.1, target_mult=100.0, max_bars=0)

    _run_bar(
        _df([100.0, 100.0], atr=3.0), baskets, pair, recovery_cfg=recovery, exit_cfg=cfg, signal=1.0
    )
    actions = _run_bar(
        _df([100.0, 90.0], atr=3.0), baskets, pair, recovery_cfg=recovery, exit_cfg=cfg, signal=1.0
    )

    assert any(a.action == ActionKind.EXIT and a.reason == ExitReason.STOP.value for a in actions)
    assert not any(a.action == ActionKind.ADD_GRID for a in actions)
    assert not any(a.reason == ExitReason.TIME.value for a in actions)
