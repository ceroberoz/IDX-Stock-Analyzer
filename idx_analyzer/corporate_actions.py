"""
Corporate Actions module for IDX Stock Analyzer.

Provides dividend history, stock splits history, and a unified corporate actions
timeline using Yahoo Finance data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yfinance as yf

from .cache import configure_yfinance_cache

configure_yfinance_cache()

logger = logging.getLogger(__name__)


def _format_ticker(ticker: str) -> str:
    """Format ticker to Yahoo Finance format."""
    if ticker.upper().startswith("IDX:"):
        ticker = ticker[4:]
    if not ticker.upper().endswith(".JK"):
        ticker = f"{ticker}.JK"
    return ticker.upper()


@dataclass
class DividendRecord:
    """A single dividend payment record."""

    date: datetime
    amount: float
    yield_on_cost: float = 0.0


@dataclass
class DividendInfo:
    """Dividend information for a stock."""

    ticker: str
    dividends: list[DividendRecord] = field(default_factory=list)
    current_yield: float = 0.0
    payout_ratio: float = 0.0
    five_year_avg_yield: float = 0.0
    ex_date: str = ""
    total_dividends_1y: float = 0.0


@dataclass
class SplitRecord:
    """A single stock split record."""

    date: datetime
    ratio: str
    ratio_value: float


@dataclass
class CorporateAction:
    """A unified corporate action entry."""

    date: datetime
    action_type: str  # "dividend", "split", "earnings"
    description: str
    value: float = 0.0


def get_dividend_info(ticker: str) -> DividendInfo:
    """
    Get dividend information for a ticker.

    Returns DividendInfo with historical dividends, yield, payout ratio,
    and other dividend-related metrics.
    """
    formatted = _format_ticker(ticker)
    clean = formatted.replace(".JK", "")
    info_result = DividendInfo(ticker=clean)

    try:
        stock = yf.Ticker(formatted)
    except Exception as e:
        logger.warning(f"Failed to create Ticker for {formatted}: {e}")
        return info_result

    # Fetch info for yield/payout data
    try:
        info = stock.info or {}
        raw_yield = float(info.get("dividendYield", 0) or 0)
        # Yahoo returns dividendYield as decimal (e.g., 0.024 = 2.4%)
        # but fiveYearAvgDividendYield as percentage (e.g., 2.38 = 2.38%)
        info_result.current_yield = raw_yield
        info_result.payout_ratio = float(info.get("payoutRatio", 0) or 0)
        info_result.five_year_avg_yield = float(
            info.get("fiveYearAvgDividendYield", 0) or 0
        )
        ex_date_ts = info.get("exDividendDate")
        if ex_date_ts:
            try:
                info_result.ex_date = datetime.fromtimestamp(ex_date_ts).strftime(
                    "%Y-%m-%d"
                )
            except (ValueError, TypeError, OSError):
                info_result.ex_date = str(ex_date_ts)
    except Exception as e:
        logger.warning(f"Failed to fetch info for {formatted}: {e}")

    # Fetch historical dividends
    try:
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            # Get price history to calculate yield on cost
            try:
                hist = stock.history(period="max")
            except Exception:
                hist = None

            records: list[DividendRecord] = []
            for date_idx, amount in dividends.items():
                div_date = date_idx.to_pydatetime()
                yoc = 0.0
                if hist is not None and not hist.empty:
                    try:
                        # Find closest price on or before dividend date
                        prior = hist.loc[:date_idx]
                        if not prior.empty:
                            price = float(prior["Close"].iloc[-1])
                            if price > 0:
                                yoc = float(amount) / price
                    except Exception as e:
                        logger.debug(f"Failed to calculate yield on cost: {e}")
                        pass
                records.append(
                    DividendRecord(
                        date=div_date,
                        amount=float(amount),
                        yield_on_cost=yoc,
                    )
                )

            info_result.dividends = records

            # Calculate total dividends in last 12 months
            one_year_ago = datetime.now() - timedelta(days=365)
            info_result.total_dividends_1y = sum(
                r.amount
                for r in records
                if r.date.replace(tzinfo=None) >= one_year_ago
            )
    except Exception as e:
        logger.warning(f"Failed to fetch dividends for {formatted}: {e}")

    return info_result


def get_splits_history(ticker: str) -> list[SplitRecord]:
    """
    Get stock split history for a ticker.

    Returns list of SplitRecord sorted by date (newest first).
    """
    formatted = _format_ticker(ticker)
    try:
        stock = yf.Ticker(formatted)
        splits = stock.splits
        if splits is None or splits.empty:
            return []

        records: list[SplitRecord] = []
        for date_idx, value in splits.items():
            ratio_val = float(value)
            # Format ratio as "N:1" (e.g., 2.0 -> "2:1")
            if ratio_val >= 1:
                ratio_str = f"{int(ratio_val)}:1"
            else:
                # Reverse split (e.g., 0.5 -> "1:2")
                denominator = int(round(1.0 / ratio_val))
                ratio_str = f"1:{denominator}"

            records.append(
                SplitRecord(
                    date=date_idx.to_pydatetime(),
                    ratio=ratio_str,
                    ratio_value=ratio_val,
                )
            )

        records.sort(key=lambda r: r.date, reverse=True)
        return records
    except Exception as e:
        logger.warning(f"Failed to fetch splits for {formatted}: {e}")
        return []


def get_corporate_actions(ticker: str) -> list[CorporateAction]:
    """
    Get a unified timeline of all corporate actions (dividends, splits, earnings).

    Returns list of CorporateAction sorted by date (newest first).
    """
    formatted = _format_ticker(ticker)
    actions: list[CorporateAction] = []

    try:
        stock = yf.Ticker(formatted)
    except Exception as e:
        logger.warning(f"Failed to create Ticker for {formatted}: {e}")
        return actions

    # Dividends and splits from actions
    try:
        stock_actions = stock.actions
        if stock_actions is not None and not stock_actions.empty:
            for date_idx, row in stock_actions.iterrows():
                action_date = date_idx.to_pydatetime()

                div_amount = float(row.get("Dividends", 0) or 0)
                split_val = float(row.get("Stock Splits", 0) or 0)

                if div_amount > 0:
                    actions.append(
                        CorporateAction(
                            date=action_date,
                            action_type="dividend",
                            description=f"Dividend payment: Rp {div_amount:,.2f}",
                            value=div_amount,
                        )
                    )

                if split_val > 0:
                    if split_val >= 1:
                        ratio_str = f"{int(split_val)}:1"
                    else:
                        denominator = int(round(1.0 / split_val))
                        ratio_str = f"1:{denominator}"
                    actions.append(
                        CorporateAction(
                            date=action_date,
                            action_type="split",
                            description=f"Stock split {ratio_str}",
                            value=split_val,
                        )
                    )
    except Exception as e:
        logger.warning(f"Failed to fetch actions for {formatted}: {e}")

    # Earnings dates
    try:
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            for date_idx, row in earnings.iterrows():
                earn_date = date_idx.to_pydatetime()
                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")

                if eps_act is not None and not (
                    isinstance(eps_act, float) and eps_act != eps_act
                ):
                    desc = f"Earnings reported: EPS Rp {float(eps_act):,.2f}"
                elif eps_est is not None and not (
                    isinstance(eps_est, float) and eps_est != eps_est
                ):
                    desc = f"Earnings expected: EPS est. Rp {float(eps_est):,.2f}"
                else:
                    desc = "Earnings date"

                actions.append(
                    CorporateAction(
                        date=earn_date,
                        action_type="earnings",
                        description=desc,
                        value=float(eps_act) if eps_act and eps_act == eps_act else 0.0,
                    )
                )
    except Exception as e:
        logger.warning(f"Failed to fetch earnings dates for {formatted}: {e}")

    actions.sort(key=lambda a: a.date, reverse=True)
    return actions


def format_dividend_report(info: DividendInfo) -> str:
    """Format DividendInfo as a readable plain text report with emojis."""
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f" 💰 DIVIDEND REPORT: {info.ticker:^42} ")
    lines.append("=" * 64)
    lines.append("")

    # Summary metrics
    lines.append("-" * 64)
    lines.append("DIVIDEND METRICS")
    lines.append("-" * 64)
    # current_yield from Yahoo is decimal (0.024 = 2.4%), display as percentage
    yield_display = info.current_yield * 100 if info.current_yield < 1 else info.current_yield
    lines.append(f"   📊 Current Yield:       {yield_display:>8.2f}%")
    lines.append(f"   📊 Payout Ratio:        {info.payout_ratio * 100:>8.2f}%")
    # fiveYearAvgDividendYield from Yahoo is already percentage
    lines.append(f"   📊 5-Year Avg Yield:    {info.five_year_avg_yield:>8.2f}%")
    lines.append(f"   💵 Total Dividends 1Y:  Rp {info.total_dividends_1y:>12,.2f}")
    if info.ex_date:
        lines.append(f"   📅 Ex-Dividend Date:    {info.ex_date}")
    lines.append("")

    # Dividend history
    if info.dividends:
        lines.append("-" * 64)
        lines.append("DIVIDEND HISTORY")
        lines.append("-" * 64)
        for rec in info.dividends[-10:]:
            date_str = rec.date.strftime("%Y-%m-%d")
            yoc_str = f"(YoC: {rec.yield_on_cost * 100:.2f}%)" if rec.yield_on_cost > 0 else ""
            lines.append(f"   💰 {date_str}  Rp {rec.amount:>12,.2f}  {yoc_str}")
        if len(info.dividends) > 10:
            lines.append(f"   ... and {len(info.dividends) - 10} more records")
        lines.append("")
    else:
        lines.append("   ℹ️  No dividend history available")
        lines.append("")

    lines.append("-" * 64)
    lines.append("Powered by Yahoo Finance")
    lines.append("")
    return "\n".join(lines)


def format_splits_report(ticker: str, splits: list[SplitRecord]) -> str:
    """Format splits history as a readable plain text report with emojis."""
    clean = ticker.upper().replace(".JK", "").replace("IDX:", "")
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f" ✂️ STOCK SPLITS HISTORY: {clean:^38} ")
    lines.append("=" * 64)
    lines.append("")

    if splits:
        lines.append("-" * 64)
        lines.append("SPLIT RECORDS")
        lines.append("-" * 64)
        for rec in splits:
            date_str = rec.date.strftime("%Y-%m-%d")
            lines.append(f"   ✂️  {date_str}  Split {rec.ratio} (ratio: {rec.ratio_value:.1f})")
        lines.append("")
        lines.append(f"   Total splits: {len(splits)}")
    else:
        lines.append("   ℹ️  No stock split history available")

    lines.append("")
    lines.append("-" * 64)
    lines.append("Powered by Yahoo Finance")
    lines.append("")
    return "\n".join(lines)


def format_corporate_actions_report(
    ticker: str, actions: list[CorporateAction]
) -> str:
    """Format corporate actions timeline as a readable plain text report with emojis."""
    clean = ticker.upper().replace(".JK", "").replace("IDX:", "")
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f" 📋 CORPORATE ACTIONS: {clean:^40} ")
    lines.append("=" * 64)
    lines.append("")

    if actions:
        icon_map = {
            "dividend": "💰",
            "split": "✂️",
            "earnings": "📊",
        }

        lines.append("-" * 64)
        lines.append("ACTIONS TIMELINE")
        lines.append("-" * 64)

        for action in actions[:20]:
            icon = icon_map.get(action.action_type, "📌")
            date_str = action.date.strftime("%Y-%m-%d")
            label = action.action_type.upper()
            lines.append(f"   {icon} [{date_str}] {label:>10}  {action.description}")

        if len(actions) > 20:
            lines.append(f"   ... and {len(actions) - 20} more actions")
        lines.append("")
        lines.append(f"   Total actions: {len(actions)}")
    else:
        lines.append("   ℹ️  No corporate actions found")

    lines.append("")
    lines.append("-" * 64)
    lines.append("Powered by Yahoo Finance")
    lines.append("")
    return "\n".join(lines)
