"""Shared labels for structural strategy semantics."""

from __future__ import annotations


class LiquidityEvent:
    BULLISH_RECLAIM = "bullish_reclaim"
    BEARISH_RECLAIM = "bearish_reclaim"
    BREAKOUT_ACCEPTANCE_HIGH = "breakout_acceptance_high"
    BREAKOUT_ACCEPTANCE_LOW = "breakout_acceptance_low"
    FAILED_BREAKOUT_HIGH = "failed_breakout_high"
    FAILED_BREAKOUT_LOW = "failed_breakout_low"
    NONE = "none"


class StructureState:
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    UNKNOWN = "unknown"
class MarketStage:
    WARMUP = "warmup"
    DATA_ERROR = "data_error"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION_OR_REVERSAL = "distribution_or_reversal"
    RANGE = StructureState.RANGE
    TREND_CONTINUATION = "trend_continuation"
    WIDE_RANGE = "wide_range"
    TRANSITION = "transition"
    UNKNOWN = StructureState.UNKNOWN


class StructureReason:
    WARMUP = "warmup_range_not_ready"
    DATA_ERROR = "data_error"
    AMBIGUOUS_TRANSITION = "ambiguous_transition"
    HIGHER_HIGH_HIGHER_LOW = "higher_high_higher_low"
    LOWER_HIGH_LOWER_LOW = "lower_high_lower_low"
    COMPRESSED_RANGE = "compressed_range"
    AMBIGUOUS_STRUCTURE = "ambiguous_structure"


class MarketStageReason:
    WARMUP = "warmup_range_not_ready"
    DATA_ERROR = "data_error"
    MARKUP_BREAKOUT = "markup_breakout"
    MARKDOWN_BREAKOUT = "markdown_breakout"
    COMPRESSED_NEAR_LOW = "compressed_near_low"
    COMPRESSED_NEAR_HIGH = "compressed_near_high"
    COMPRESSED_MID_RANGE = "compressed_mid_range"
    TREND_WITHOUT_RANGE_BREAK = "trend_without_range_break"
    WIDE_RANGE_NO_STAGE = "wide_range_no_stage"
    AMBIGUOUS_TRANSITION = "ambiguous_transition"
    UNKNOWN_UNHANDLED = "unknown_unhandled"


class StageUnknownReason:
    WARMUP = "warmup"
    WIDE_RANGE = "wide_range"
    TRANSITION = "transition"
    DATA_ERROR = "data_error"
    NONE = "none"


class StateKeyColumn:
    MTF_STATE_KEY = "mtf_state_key"
    MTF_STRUCTURE_KEY = "mtf_structure_key"
    MTF_STAGE_KEY = "mtf_stage_key"
    MTF_EVENT_STATE_KEY = "mtf_event_state_key"


class ClassifierColumn:
    STRUCTURE_TREND_STATE = "structure_trend_state"
    MARKET_STAGE = "market_stage"
    STRUCTURE_REASON = "structure_reason"
    MARKET_STAGE_REASON = "market_stage_reason"
    STAGE_UNKNOWN_REASON = "stage_unknown_reason"
    RANGE_WIDTH_ATR = "range_width_atr"
    RANGE_WIDTH_ATR_THRESHOLD = "range_width_atr_threshold"
    RANGE_WIDTH_THRESHOLD_MODE = "range_width_threshold_mode"
    RANGE_WIDTH_THRESHOLD_READY = "range_width_threshold_ready"
    RANGE_WIDTH_THRESHOLD_SOURCE = "range_width_threshold_source"


class ClassifierDiagnosticName:
    COVERAGE = "Classifier coverage"
    STAGE_DISTRIBUTION = "Stage distribution"
    TREND_DISTRIBUTION = "Trend distribution"
    REASON_DISTRIBUTION = "Reason distribution"
    UNKNOWN_REASON_CONSISTENCY = "Unknown reason consistency"
    RESOLVED_NONE_AUDIT = "Resolved none audit"
    RAW_UNKNOWN_ATTRIBUTION = "Raw unknown attribution"
    THRESHOLD_DISTRIBUTION = "Threshold distribution"
    STRUCTURE_STAGE_MATRIX = "Structure x stage matrix"
    STAGE_REASON_MATRIX = "Stage x reason matrix"
    MTF_STATE_CARDINALITY = "MTF state cardinality"
    MTF_STATE_TRANSITION_SUMMARY = "MTF state transition summary"
    MTF_STATE_TRANSITION_MATRIX = "MTF state transition matrix"
    MTF_STATE_DWELL_DISTRIBUTION = "MTF state dwell distribution"
    MTF_STATE_TIME_DISTRIBUTION = "MTF state time distribution"
    MTF_RIGHT_EDGE_DRIFT = "MTF right-edge drift"


class DiagnosticColumn:
    ENTRY_LIQUIDITY_EVENT_TYPE = "entry_liquidity_event_type"
    ENTRY_LIQUIDITY_EVENT_TYPE_BUCKET = "entry_liquidity_event_type_bucket"
    ENTRY_EVENT_QUALITY_SCORE = "entry_event_quality_score"
    ENTRY_EVENT_QUALITY_BUCKET = "entry_event_quality_bucket"


class LossCause:
    ACCEPTED_BREAKOUT_AGAINST_REVERSION = "accepted_breakout_against_reversion"
    RECLAIM_FAILED = "reclaim_failed"
    FAILED_BREAKOUT = "failed_breakout"
    VOLATILITY_EXPANSION = "volatility_expansion"
    TREND_CONTINUATION_AGAINST_REVERSION = "trend_continuation_against_reversion"
    UNCLASSIFIED_NONE_EVENT = "unclassified_none_event"
    STOP_NO_REVERSION = "stop_no_reversion"
    EXIT_MISMATCH_OR_NO_REVERSION = "exit_mismatch_or_no_reversion"
    UNCLASSIFIED = "unclassified"
