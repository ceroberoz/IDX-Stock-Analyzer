"""
Technical Stock Screener for IDX Analyzer.

Screen Indonesian stocks based on technical analysis criteria.
Uses Yahoo Finance API for data.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from .analyzer import IDXAnalyzer
from .sectors_data import LIQUID_UNIVERSE


class FilterOperator(Enum):
    """Comparison operators for filters."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "=="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    BETWEEN = "between"


@dataclass
class ScreenerFilter:
    """Single screener filter criteria."""

    name: str
    operator: FilterOperator
    value: Union[float, tuple]
    field: str  # Which AnalysisResult field to check

    def check(self, value: float) -> bool:
        """Check if value passes this filter."""
        if self.operator == FilterOperator.GREATER_THAN:
            return value > self.value
        elif self.operator == FilterOperator.LESS_THAN:
            return value < self.value
        elif self.operator == FilterOperator.EQUAL:
            return value == self.value
        elif self.operator == FilterOperator.GREATER_EQUAL:
            return value >= self.value
        elif self.operator == FilterOperator.LESS_EQUAL:
            return value <= self.value
        elif self.operator == FilterOperator.BETWEEN:
            low, high = self.value
            return low <= value <= high
        return False


@dataclass
class ScreenerResult:
    """Result from screening a single stock."""

    ticker: str
    passed: bool
    current_price: float = 0.0
    change_percent: float = 0.0
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    trend: str = ""
    volume: int = 0
    failed_filters: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ScreenerConfig:
    """Configuration for stock screening."""

    filters: List[ScreenerFilter] = field(default_factory=list)
    max_workers: int = 3  # Concurrent workers for API calls
    delay_between_calls: float = 0.5  # Rate limiting
    timeout_per_stock: int = 30
    min_data_points: int = 50  # Minimum days of data required


class TechnicalScreener:
    """
    Technical stock screener for Indonesian stocks.

    Screen stocks based on technical indicators like RSI, SMA crossovers,
    MACD signals, trend direction, and more.
    """

    # Default list of liquid IDX stocks to screen
    # Uses comprehensive liquid universe from sectors_data
    DEFAULT_UNIVERSE = LIQUID_UNIVERSE

    def __init__(
        self,
        config: Optional[ScreenerConfig] = None,
        universe: Optional[List[str]] = None,
    ):
        self.config = config or ScreenerConfig()
        self.universe = universe or self.DEFAULT_UNIVERSE
        self.results: List[ScreenerResult] = []

    def _analyze_single_stock(self, ticker: str) -> ScreenerResult:
        """Analyze a single stock against all filters."""
        try:
            analyzer = IDXAnalyzer(ticker)
            analyzer.fetch_data(period="6mo")

            if (
                analyzer.hist is None
                or len(analyzer.hist) < self.config.min_data_points
            ):
                return ScreenerResult(
                    ticker=ticker,
                    passed=False,
                    error=f"Insufficient data ({len(analyzer.hist) if analyzer.hist else 0} points)",
                )

            result = analyzer.analyze()
            failed_filters = []

            # Check each filter
            for filter_criteria in self.config.filters:
                value = getattr(result, filter_criteria.field, None)
                if value is None:
                    failed_filters.append(f"{filter_criteria.name}: N/A")
                    continue

                if not filter_criteria.check(value):
                    failed_filters.append(
                        f"{filter_criteria.name}: {value:.2f} (required: {filter_criteria.operator.value} {filter_criteria.value})"
                    )

            return ScreenerResult(
                ticker=ticker,
                passed=len(failed_filters) == 0,
                current_price=result.current_price,
                change_percent=result.change_percent,
                rsi=result.rsi,
                sma_20=result.sma_20,
                sma_50=result.sma_50,
                sma_200=result.sma_200,
                macd_line=result.macd_line,
                macd_signal=result.macd_signal,
                macd_histogram=result.macd_histogram,
                trend=result.trend,
                volume=result.volume,
                failed_filters=failed_filters,
            )

        except Exception as e:
            return ScreenerResult(
                ticker=ticker,
                passed=False,
                error=str(e),
            )

    def screen(
        self,
        tickers: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ScreenerResult]:
        """
        Screen stocks against configured filters.

        Args:
            tickers: List of tickers to screen (uses universe if None)
            progress_callback: Called with (completed, total) for progress updates

        Returns:
            List of ScreenerResult for all screened stocks
        """
        tickers_to_screen = tickers or self.universe
        self.results = []
        total = len(tickers_to_screen)

        print(f"Screening {total} stocks...")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._analyze_single_stock, ticker): ticker
                for ticker in tickers_to_screen
            }

            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_stock)
                    self.results.append(result)
                except Exception as e:
                    self.results.append(
                        ScreenerResult(
                            ticker=ticker,
                            passed=False,
                            error=f"Timeout or error: {e}",
                        )
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

                # Rate limiting
                time.sleep(self.config.delay_between_calls)

        return self.results

    def get_passed_stocks(self) -> List[ScreenerResult]:
        """Get only stocks that passed all filters."""
        return [r for r in self.results if r.passed]

    def to_dataframe(self, only_passed: bool = False) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = self.get_passed_stocks() if only_passed else self.results

        if not data:
            return pd.DataFrame()

        rows = []
        for r in data:
            row = {
                "Ticker": r.ticker,
                "Passed": "Yes" if r.passed else "No",
                "Price": r.current_price,
                "Change %": r.change_percent,
                "RSI": r.rsi,
                "SMA 20": r.sma_20,
                "SMA 50": r.sma_50,
                "Trend": r.trend.split("|")[0].strip() if r.trend else "",
                "Volume": r.volume,
            }
            if r.error:
                row["Error"] = r.error
            elif r.failed_filters:
                row["Failed Filters"] = "; ".join(r.failed_filters)
            rows.append(row)

        return pd.DataFrame(rows)

    def print_results(
        self,
        only_passed: bool = True,
        max_rows: Optional[int] = None,
    ) -> None:
        """Print screening results to console."""
        df = self.to_dataframe(only_passed=only_passed)

        if df.empty:
            print("No stocks match the criteria.")
            return

        if max_rows:
            df = df.head(max_rows)

        print("\n" + "=" * 100)
        print(f"TECHNICAL SCREENER RESULTS")
        print("=" * 100)

        if only_passed:
            print(f"Found {len(df)} stocks matching all criteria\n")
        else:
            passed = len(self.get_passed_stocks())
            print(f"Passed: {passed}/{len(self.results)} stocks\n")

        # Format and print
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 30)

        print(df.to_string(index=False))
        print("=" * 100)


# ============================================================================
# PRESET SCREENERS
# ============================================================================


def create_oversold_screener() -> TechnicalScreener:
    """Screener for oversold stocks (RSI < 30)."""
    config = ScreenerConfig(
        filters=[
            ScreenerFilter(
                name="RSI Oversold",
                operator=FilterOperator.LESS_THAN,
                value=30.0,
                field="rsi",
            ),
        ]
    )
    return TechnicalScreener(config)


def create_overbought_screener() -> TechnicalScreener:
    """Screener for overbought stocks (RSI > 70)."""
    config = ScreenerConfig(
        filters=[
            ScreenerFilter(
                name="RSI Overbought",
                operator=FilterOperator.GREATER_THAN,
                value=70.0,
                field="rsi",
            ),
        ]
    )
    return TechnicalScreener(config)


def create_bullish_trend_screener() -> TechnicalScreener:
    """Screener for bullish trend (Price > SMA 50)."""
    config = ScreenerConfig(
        filters=[
            ScreenerFilter(
                name="Above SMA 50",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field="sma_50",
            ),
        ]
    )
    return TechnicalScreener(config)


def create_macd_bullish_screener() -> TechnicalScreener:
    """Screener for MACD bullish crossover (Histogram > 0)."""
    config = ScreenerConfig(
        filters=[
            ScreenerFilter(
                name="MACD Bullish",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field="macd_histogram",
            ),
        ]
    )
    return TechnicalScreener(config)


def create_strong_buy_screener() -> TechnicalScreener:
    """
    Screener for strong buy signals:
    - RSI between 30-50 (recovering from oversold)
    - Price above SMA 20 (short-term uptrend)
    - MACD histogram positive
    """
    config = ScreenerConfig(
        filters=[
            ScreenerFilter(
                name="RSI Recovery",
                operator=FilterOperator.BETWEEN,
                value=(30.0, 50.0),
                field="rsi",
            ),
            ScreenerFilter(
                name="Above SMA 20",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field="sma_20",
            ),
            ScreenerFilter(
                name="MACD Positive",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field="macd_histogram",
            ),
        ]
    )
    return TechnicalScreener(config)


def create_golden_cross_screener() -> TechnicalScreener:
    """Screener for Golden Cross setup (SMA 50 > SMA 200)."""
    # This requires special handling since we compare two SMAs
    # We'll filter after analysis
    config = ScreenerConfig()
    screener = TechnicalScreener(config)
    return screener


# ============================================================================
# CLI HELPER FUNCTIONS
# ============================================================================


def build_screener_from_args(args) -> TechnicalScreener:
    """Build a TechnicalScreener from CLI arguments."""
    filters = []

    # RSI filters
    if args.rsi_below:
        filters.append(
            ScreenerFilter(
                name=f"RSI < {args.rsi_below}",
                operator=FilterOperator.LESS_THAN,
                value=args.rsi_below,
                field="rsi",
            )
        )
    if args.rsi_above:
        filters.append(
            ScreenerFilter(
                name=f"RSI > {args.rsi_above}",
                operator=FilterOperator.GREATER_THAN,
                value=args.rsi_above,
                field="rsi",
            )
        )

    # SMA filters
    if args.above_sma:
        field = f"sma_{args.above_sma}"
        filters.append(
            ScreenerFilter(
                name=f"Price > SMA {args.above_sma}",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field=field,
            )
        )
    if args.below_sma:
        field = f"sma_{args.below_sma}"
        filters.append(
            ScreenerFilter(
                name=f"Price < SMA {args.below_sma}",
                operator=FilterOperator.LESS_THAN,
                value=0.0,
                field=field,
            )
        )

    # MACD filter
    if args.macd_bullish:
        filters.append(
            ScreenerFilter(
                name="MACD Bullish",
                operator=FilterOperator.GREATER_THAN,
                value=0.0,
                field="macd_histogram",
            )
        )
    if args.macd_bearish:
        filters.append(
            ScreenerFilter(
                name="MACD Bearish",
                operator=FilterOperator.LESS_THAN,
                value=0.0,
                field="macd_histogram",
            )
        )

    # Price change filter
    if args.change_above:
        filters.append(
            ScreenerFilter(
                name=f"Change > {args.change_above}%",
                operator=FilterOperator.GREATER_THAN,
                value=args.change_above,
                field="change_percent",
            )
        )
    if args.change_below:
        filters.append(
            ScreenerFilter(
                name=f"Change < {args.change_below}%",
                operator=FilterOperator.LESS_THAN,
                value=args.change_below,
                field="change_percent",
            )
        )

    config = ScreenerConfig(
        filters=filters,
        max_workers=args.screener_workers or 3,
    )

    # Handle universe
    universe = None
    if args.screener_sector:
        # Filter universe by sector
        from .sectors_data import get_tickers_by_sector

        universe = get_tickers_by_sector(args.screener_sector)
    elif args.screener_tickers:
        universe = [t.strip().upper() for t in args.screener_tickers.split(",")]

    return TechnicalScreener(config, universe)
