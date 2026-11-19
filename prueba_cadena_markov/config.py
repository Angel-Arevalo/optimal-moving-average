from dataclasses import dataclass, field

FEATURES = ("distance_to_ma", "distance_change", "ma_slope",
            "realized_volatility", "efficiency_ratio")


@dataclass(frozen=True)
class FeatureConfig:
    slope_period: int = 3
    volatility_period: int = 20
    efficiency_period: int = 20

    def __post_init__(self):
        if min(self.slope_period, self.volatility_period, self.efficiency_period) < 1:
            raise ValueError("Feature periods must be positive")


@dataclass(frozen=True)
class MarkovConfig:
    enabled: bool = False
    n_components: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    probability_threshold: float = .70
    min_trades_per_regime: int = 30
    min_observations: int = 300
    min_state_observations: int = 30
    covariance_floor: float = 1e-6
    max_condition: float = 1e8
    bootstrap_samples: int = 2000
    bootstrap_block: int = 5
    confidence: float = .95
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        if self.covariance_type != "diag":
            raise ValueError("This implementation validates diagonal covariances only")
        if not 0 < self.probability_threshold <= 1 or not 0 < self.confidence < 1:
            raise ValueError("Invalid probability/confidence")
        if min(self.n_components, self.n_iter, self.min_trades_per_regime,
               self.min_observations, self.min_state_observations,
               self.bootstrap_samples, self.bootstrap_block) < 1:
            raise ValueError("Counts must be positive")
        if self.tol <= 0 or self.covariance_floor <= 0 or self.max_condition <= 1:
            raise ValueError("Invalid numerical controls")


@dataclass(frozen=True)
class RunConfig:
    # Four disjoint blocks. MA is chosen BEFORE the HMM fit observations.
    optimization_weeks: int = 26
    fit_weeks: int = 26
    calibration_weeks: int = 52
    evaluation_weeks: int = 8
    optimizer_calls: int = 20
    candle_max: int = 6
    lookback_max: int = 40
    methods: tuple = ("SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT")
    temporality: str = "h"
    # tester.py has no capital/volume engine: one unit reproduces its price P&L.
    initial_capital: float = 100000.
    units: float = 1.
    fee_bps: float = 0.

    def __post_init__(self):
        if min(self.optimization_weeks, self.fit_weeks, self.calibration_weeks,
               self.evaluation_weeks, self.candle_max) < 1:
            raise ValueError("Window lengths and candles must be positive")
        if self.optimizer_calls < 10 or self.lookback_max < 2:
            raise ValueError("Original optimizer needs >=10 calls and lookback >=2")
        if self.initial_capital <= 0 or self.units <= 0 or self.fee_bps < 0:
            raise ValueError("Invalid accounting parameters")
        if self.temporality not in ("min", "h", "d"):
            raise ValueError("Unsupported temporality")
