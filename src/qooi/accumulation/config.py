"""Strict TOML-backed config for the accumulation scanner."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator

from qooi.core.config import StrictConfigModel

UniverseName = Literal["core", "research"]
DataSource = Literal["swap", "spot_signal_swap_exec", "spot"]
ChainName = Literal["ethereum", "bsc"]
OnchainProvider = Literal["etherscan"]
BookMode = Literal["snapshot", "sample", "off"]
PolymarketProvider = Literal["gamma"]
BroadProvider = Literal["coingecko", "coinpaprika", "defillama", "cryptopanic"]


class RunConfig(StrictConfigModel):
    universe: UniverseName = "research"
    out: str = "data/output/accumulation/mvp"

    @field_validator("out")
    @classmethod
    def _out_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run.out must not be empty")
        return value


class MarketConfig(StrictConfigModel):
    ds: DataSource = "swap"
    bar: str = "1H"
    days: int = Field(default=60, gt=0)
    refresh: bool = False
    book_samples: int = Field(default=720, ge=0)
    book_every_seconds: float = Field(default=5.0, ge=0.0)
    book_depth: int = Field(default=25, gt=0)
    book_mode: BookMode = "snapshot"
    book_snapshot_count: int = Field(default=1, ge=0)
    book_sample_symbols_max: int = Field(default=5, ge=0)
    book_sample_total_seconds: float = Field(default=0.0, ge=0.0)


class DiscoveryConfig(StrictConfigModel):
    top_n: int = Field(default=25, gt=0)
    min_volume_usd: float = Field(default=1_000_000.0, ge=0.0)
    max_spread_bps: float = Field(default=50.0, gt=0.0)
    min_history_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    missing_contract_penalty: float = Field(default=2.0, ge=0.0)
    spread_bps_penalty_scale: float = Field(default=100.0, gt=0.0)
    coverage_bonus_scale: float = Field(default=100.0, gt=0.0)


class BroadCoinGeckoConfig(StrictConfigModel):
    api_key_env: str = "COINGECKO_DEMO_API_KEY"
    vs_currency: str = "usd"
    order: str = "volume_desc"
    per_page: int = Field(default=250, gt=0, le=250)
    price_change_percentage: tuple[str, ...] = ("1h", "24h")
    include_trending: bool = True
    trending_weight: float = Field(default=4.0, ge=0.0)


class BroadCryptoPanicConfig(StrictConfigModel):
    api_key_env: str = "CRYPTOPANIC_API_KEY"
    enabled_without_key: bool = False
    limit: int = Field(default=100, gt=0, le=100)


class BroadScanConfig(StrictConfigModel):
    providers: tuple[BroadProvider, ...] = ("coingecko", "coinpaprika", "defillama")
    optional_providers: tuple[BroadProvider, ...] = ("cryptopanic",)
    max_assets: int = Field(default=500, gt=0)
    coingecko_pages: int = Field(default=2, gt=0, le=10)
    coinpaprika_max_assets: int = Field(default=1000, gt=0)
    min_market_cap_usd: float = Field(default=1_000_000.0, ge=0.0)
    max_market_cap_usd: float = Field(default=500_000_000.0, ge=0.0)
    min_volume_24h_usd: float = Field(default=500_000.0, ge=0.0)
    tvl_change_1d_min_pct: float = 20.0
    output_top_n: int = Field(default=25, gt=0)
    excluded_base_ccy: tuple[str, ...] = ("USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE")
    coingecko: BroadCoinGeckoConfig = Field(default_factory=BroadCoinGeckoConfig)
    cryptopanic: BroadCryptoPanicConfig = Field(default_factory=BroadCryptoPanicConfig)


class PolymarketAliasConfig(StrictConfigModel):
    symbol: str
    queries: tuple[str, ...]

    @field_validator("symbol")
    @classmethod
    def _symbol_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sources.polymarket.aliases.symbol must not be empty")
        return value

    @field_validator("queries")
    @classmethod
    def _queries_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(query.strip() for query in value if query.strip())
        if not cleaned:
            raise ValueError("sources.polymarket.aliases.queries must not be empty")
        return cleaned


class PolymarketSourceConfig(StrictConfigModel):
    provider: PolymarketProvider = "gamma"
    search_limit_per_symbol: int = Field(default=10, gt=0, le=100)
    max_markets_per_symbol: int = Field(default=25, gt=0, le=250)
    min_volume_usd: float = Field(default=0.0, ge=0.0)
    include_closed: bool = False
    lookback_hours: int = Field(default=168, gt=0)
    fetch_concurrency: int = Field(default=3, gt=0)
    aliases: tuple[PolymarketAliasConfig, ...] = ()


class LocalMessageSourceConfig(StrictConfigModel):
    path: str = ""
    default_source: str = "local_csv"


class DisabledSourceConfig(StrictConfigModel):
    families: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    polymarket_queries: tuple[str, ...] = ()


class SourceConfig(StrictConfigModel):
    fetch_concurrency: int = Field(default=3, gt=0)
    trade_limit: int = Field(default=100, gt=0, le=500)
    funding_limit: int = Field(default=100, gt=0, le=400)
    open_interest_refresh: bool = False
    rubik_period: str = "1H"
    rubik_limit: int = Field(default=100, gt=0, le=100)
    rubik_taker_unit: Literal["0", "1", "2"] = "2"
    disabled: DisabledSourceConfig = Field(default_factory=DisabledSourceConfig)
    polymarket: PolymarketSourceConfig = Field(default_factory=PolymarketSourceConfig)
    messages: LocalMessageSourceConfig = Field(default_factory=LocalMessageSourceConfig)


class SummaryConfig(StrictConfigModel):
    top_n: int = Field(default=10, gt=0)
    latest_only: bool = True


class PotentialScanConfig(StrictConfigModel):
    enabled: bool = True
    output_top_n: int = Field(default=40, gt=0)
    min_market_cap_usd: float = Field(default=1_000_000.0, ge=0.0)
    max_market_cap_usd: float = Field(default=500_000_000.0, ge=0.0)
    min_volume_24h_usd: float = Field(default=500_000.0, ge=0.0)
    near_low_ratio: float = Field(default=1.30, gt=0.0)
    strong_near_low_ratio: float = Field(default=1.10, gt=0.0)
    compression_pctile_max: float = Field(default=0.30, ge=0.0, le=1.0)
    volume_contraction_max: float = Field(default=0.60, ge=0.0)
    volume_spike_ratio: float = Field(default=3.0, gt=0.0)
    first_spike_lookback_hours: int = Field(default=120, gt=0)
    range_low_max_pct: float = Field(default=0.35, ge=0.0, le=1.0)
    late_range_min_pct: float = Field(default=0.80, ge=0.0, le=1.0)
    taker_buy_confirm_min: float = Field(default=0.65, ge=0.0, le=1.0)
    min_history_hours: int = Field(default=720, gt=0)
    full_history_hours: int = Field(default=2160, gt=0)
    min_base_duration_hours: int = Field(default=168, gt=0)
    max_new_low_count_30d: int = Field(default=2, ge=0)
    max_ma30_down_slope: float = -0.03
    ma7_reclaim_min_pct: float = 0.0
    ma30_reclaim_min_pct: float = 0.0
    strong_downtrend_60d_return: float = -0.35
    strong_downtrend_90d_return: float = -0.50


class DatabaseConfig(StrictConfigModel):
    enabled: bool = False
    path: str = "db/accumulation.sqlite"

    @field_validator("path")
    @classmethod
    def _path_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("database.path must not be empty")
        return value


class TokenConfig(StrictConfigModel):
    symbol: str
    chain: ChainName
    token_address: str

    @field_validator("symbol", "token_address")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("token entries require non-empty symbol and token_address")
        return value


class OnchainConfig(StrictConfigModel):
    provider: OnchainProvider = "etherscan"
    lookback_days: int = Field(default=60, gt=0)
    poll_minutes: int = Field(default=10, gt=0)
    exchange_address_book: str = ""
    tokens: tuple[TokenConfig, ...] = ()


class FeatureConfig(StrictConfigModel):
    flow_zscore_window_hours: int = Field(default=168, gt=1)
    flow_negative_streak_hours: int = Field(default=2, gt=0)
    depth_window_minutes: int = Field(default=60, gt=0)
    large_trade_usd: float = Field(default=50_000.0, gt=0.0)
    large_trade_multiple: float = Field(default=5.0, gt=0.0)
    resilience_minutes: int = Field(default=5, gt=0)
    ma_hours: int = Field(default=200, gt=1)
    max_source_staleness_hours: int = Field(default=48, gt=0)


class ScoringConfig(StrictConfigModel):
    yellow_threshold: int = 20
    orange_threshold: int = 35
    red_threshold: int = 40
    flow_outflow_z: float = -3.0
    flow_inflow_z: float = 3.0
    whale_accumulation_threshold: float = 0.70
    depth_imbalance_threshold: float = 0.30
    down_day_return_threshold: float = -0.02
    resilience_threshold: float = 0.60
    mention_growth_hot: float = 3.0
    max_position_pct: float = 0.05
    stop_loss_pct: float = 0.05

    @field_validator(
        "flow_outflow_z",
        "flow_inflow_z",
        "whale_accumulation_threshold",
        "depth_imbalance_threshold",
        "down_day_return_threshold",
        "resilience_threshold",
        "mention_growth_hot",
        "max_position_pct",
        "stop_loss_pct",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("scoring thresholds must be finite")
        return value

    @model_validator(mode="after")
    def _threshold_order(self) -> ScoringConfig:
        if not (self.yellow_threshold <= self.orange_threshold <= self.red_threshold):
            raise ValueError("alert thresholds must be ordered yellow <= orange <= red")
        return self


class AccumulationConfig(StrictConfigModel):
    run: RunConfig = Field(default_factory=RunConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    onchain: OnchainConfig = Field(default_factory=OnchainConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    broad_scan: BroadScanConfig = Field(default_factory=BroadScanConfig)
    sources: SourceConfig = Field(default_factory=SourceConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    potential_scan: PotentialScanConfig = Field(default_factory=PotentialScanConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.run.out)


def load_accumulation_config(path: Path) -> AccumulationConfig:
    env_path = path.parent / ".env"
    load_dotenv(env_path if env_path.exists() else None, override=False)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return AccumulationConfig.model_validate(data)
