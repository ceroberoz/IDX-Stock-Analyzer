"""
Command-line interface for IDX Analyzer
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .analyzer import IDXAnalyzer
from .config import Config, create_default_config, load_config
from .exceptions import (
    AnalysisError,
    ChartError,
    IDXAnalyzerError,
    InsufficientDataError,
    InvalidTickerError,
    NetworkError,
    format_error_for_user,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="idx-analyzer",
        description="""
 IDX Stock Analyzer v1.0.0
 Indonesian Stock Market Technical Analysis Tool

Analyze Indonesian stocks (IDX) and get support/resistance levels,
trend analysis, and trading recommendations.

Examples:
  idx-analyzer BBCA                    # Analyze BBCA
  idx-analyzer IDX:BBCA                # With IDX: prefix
  idx-analyzer BBCA --period 1y        # 1 year of data
  idx-analyzer TLKM --export csv       # Export to CSV
  idx-analyzer BBCA --chart            # Generate chart
  idx-analyzer ASII -c -p 1y           # Chart with 1 year data
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "ticker", help="Stock ticker (e.g., BBCA, TLKM, ASII, or IDX:BBCA)"
    )

    parser.add_argument(
        "-p",
        "--period",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        help="Historical data period (default: from config or 6mo)",
    )

    parser.add_argument(
        "-e", "--export", choices=["csv", "json"], help="Export analysis to file format"
    )

    parser.add_argument("-o", "--output", help="Output file path for export")

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (useful for scripting)",
    )

    parser.add_argument(
        "-c",
        "--chart",
        action="store_true",
        help="Generate technical analysis chart (PNG)",
    )

    parser.add_argument(
        "--chart-output",
        help="Custom chart output filename (default: TICKER_chart.png)",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Generate compact output for Telegram/WhatsApp",
    )

    parser.add_argument(
        "--config",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create default configuration file and exit",
    )

    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Analyze news sentiment using Yahoo Finance + FinBERT (requires: pip install transformers torch)",
    )

    parser.add_argument(
        "--sentiment-vader",
        action="store_true",
        help="Use lightweight VADER sentiment (no model download, faster)",
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    return parser


def format_output(result, quiet: bool = False) -> str:
    """Format analysis output"""
    if quiet:
        lines = [
            f"TICKER:{result.ticker}",
            f"PRICE:{result.current_price:.0f}",
            f"CHANGE:{result.change_percent:.2f}",
            f"TREND:{result.trend}",
        ]
        if result.support_levels:
            lines.append(f"SUPPORT:{result.support_levels[0].level:.0f}")
        if result.resistance_levels:
            lines.append(f"RESISTANCE:{result.resistance_levels[0].level:.0f}")
        return "\n".join(lines)

    output = []
    output.append("")
    output.append("=" * 64)
    output.append(f" IDX Stock Analysis: {result.ticker:^42} ")
    output.append("=" * 64)
    output.append("")

    change_color = "+" if result.change_percent >= 0 else "-"
    output.append(f"Current Price: {result.current_price:>12,.0f} IDR")
    output.append(f"   Daily Change:  {change_color} {result.change_percent:>+10.2f}%")
    output.append(f"   Volume:        {result.volume:>12,}")
    output.append("")

    output.append("-" * 64)
    output.append("52-WEEK RANGE")
    output.append("-" * 64)
    from_high = (result.current_price - result.week_52_high) / result.week_52_high * 100
    from_low = (result.current_price - result.week_52_low) / result.week_52_low * 100
    output.append(
        f"   High: {result.week_52_high:>10,.0f}  ({from_high:+.1f}% from current)"
    )
    output.append(
        f"   Low:  {result.week_52_low:>10,.0f}  (+{from_low:.1f}% from current)"
    )
    output.append("")

    output.append("-" * 64)
    output.append("SUPPORT LEVELS (Buy Zones)")
    output.append("-" * 64)
    if result.support_levels:
        for i, s in enumerate(result.support_levels[:4], 1):
            dist = (result.current_price - s.level) / result.current_price * 100
            strength_icon = "*" if s.strength == "strong" else "-"
            output.append(
                f"   {i}. {s.level:>8,.0f}  ({dist:>5.1f}% below)  {strength_icon} {s.strength}"
            )
    else:
        output.append("   No clear support levels identified")
    output.append("")

    output.append("-" * 64)
    output.append("RESISTANCE LEVELS (Sell/Target Zones)")
    output.append("-" * 64)
    if result.resistance_levels:
        for i, r in enumerate(result.resistance_levels[:4], 1):
            dist = (r.level - result.current_price) / result.current_price * 100
            strength_icon = "*" if r.strength == "strong" else "-"
            output.append(
                f"   {i}. {r.level:>8,.0f}  (+{dist:>5.1f}% above)  {strength_icon} {r.strength}"
            )
    else:
        output.append("   No clear resistance levels identified")
    output.append("")

    output.append("-" * 64)
    output.append("MOVING AVERAGES")
    output.append("-" * 64)

    sma_20_pct = (result.current_price - result.sma_20) / result.sma_20 * 100
    sma_50_pct = (result.current_price - result.sma_50) / result.sma_50 * 100

    icon_20 = ">" if result.current_price > result.sma_20 else "<"
    icon_50 = ">" if result.current_price > result.sma_50 else "<"

    output.append(f"   SMA 20: {result.sma_20:>10,.0f}  {icon_20} {sma_20_pct:+.1f}%")
    output.append(f"   SMA 50: {result.sma_50:>10,.0f}  {icon_50} {sma_50_pct:+.1f}%")

    if result.sma_200:
        sma_200_pct = (result.current_price - result.sma_200) / result.sma_200 * 100
        icon_200 = ">" if result.current_price > result.sma_200 else "<"
        output.append(
            f"   SMA 200: {result.sma_200:>9,.0f}  {icon_200} {sma_200_pct:+.1f}%"
        )

    output.append("")

    if result.bb_middle is not None:
        output.append("-" * 64)
        output.append("BOLLINGER BANDS (20, 2)")
        output.append("-" * 64)
        bb_position_text = {
            "above_upper": "Above Upper (Overbought)",
            "near_upper": "Near Upper (Extended)",
            "middle": "Middle (Neutral)",
            "near_lower": "Near Lower (Oversold)",
            "below_lower": "Below Lower (Oversold)",
        }.get(result.bb_position, "Middle")
        output.append(f"   Upper:  {result.bb_upper:>10,.0f}")
        output.append(f"   Middle: {result.bb_middle:>10,.0f}  {bb_position_text}")
        output.append(f"   Lower:  {result.bb_lower:>10,.0f}")
        output.append("")

    if result.vp_poc is not None:
        output.append("-" * 64)
        output.append("VOLUME PROFILE")
        output.append("-" * 64)
        output.append(f"   POC (Point of Control): {result.vp_poc:>10,.0f}")
        output.append(f"   Value Area High:        {result.vp_value_area_high:>10,.0f}")
        output.append(f"   Value Area Low:         {result.vp_value_area_low:>10,.0f}")

        if result.vp_value_area_high and result.vp_value_area_low:
            price_in_va = (
                result.vp_value_area_low
                <= result.current_price
                <= result.vp_value_area_high
            )
            va_status = "Inside Value Area" if price_in_va else "Outside Value Area"
            va_icon = ">" if price_in_va else "~"
            output.append(f"   Current Position:       {va_icon} {va_status}")
        output.append("")

    output.append("-" * 64)
    trend_icon = (
        "BULL"
        if "Bull" in result.trend
        else "BEAR"
        if "Bear" in result.trend
        else "NEUTRAL"
    )
    output.append(f"Trend: {trend_icon} {result.trend}")
    output.append("")
    output.append("RECOMMENDATION:")
    output.append(f"   {result.recommendation}")
    output.append("")
    output.append("-" * 64)
    output.append("")

    return "\n".join(output)


def export_to_csv(result, filepath: str):
    """Export analysis to CSV"""
    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Level", "Distance %", "Strength", "Description"])
        writer.writerow(["Current", result.current_price, "", "", ""])

        for s in result.support_levels:
            dist = (result.current_price - s.level) / result.current_price * 100
            writer.writerow(
                ["Support", s.level, f"{dist:.2f}", s.strength, s.description]
            )

        for r in result.resistance_levels:
            dist = (r.level - result.current_price) / result.current_price * 100
            writer.writerow(
                ["Resistance", r.level, f"{dist:.2f}", r.strength, r.description]
            )

    print(f"Exported to: {filepath}")


def export_to_json(result, filepath: str):
    """Export analysis to JSON"""
    import json

    data = {
        "ticker": result.ticker,
        "current_price": result.current_price,
        "change_percent": result.change_percent,
        "volume": result.volume,
        "week_52_high": result.week_52_high,
        "week_52_low": result.week_52_low,
        "trend": result.trend,
        "recommendation": result.recommendation,
        "support_levels": [
            {
                "level": s.level,
                "strength": s.strength,
                "description": s.description,
                "distance_pct": round(
                    (result.current_price - s.level) / result.current_price * 100, 2
                ),
            }
            for s in result.support_levels
        ],
        "resistance_levels": [
            {
                "level": r.level,
                "strength": r.strength,
                "description": r.description,
                "distance_pct": round(
                    (r.level - result.current_price) / result.current_price * 100, 2
                ),
            }
            for r in result.resistance_levels
        ],
        "bollinger_bands": {
            "middle": result.bb_middle,
            "upper": result.bb_upper,
            "lower": result.bb_lower,
            "position": result.bb_position,
        }
        if result.bb_middle is not None
        else None,
        "volume_profile": {
            "poc": result.vp_poc,
            "value_area_high": result.vp_value_area_high,
            "value_area_low": result.vp_value_area_low,
            "total_volume": result.vp_total_volume,
        }
        if result.vp_poc is not None
        else None,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported to: {filepath}")


def main(args: Optional[list] = None) -> int:
    """Main entry point"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.init_config:
        try:
            config_path = create_default_config()
            print(f"Created default config at: {config_path}")
            return 0
        except Exception as e:
            print(f"Error creating config: {e}", file=sys.stderr)
            return 1

    config: Optional[Config] = None
    if parsed_args.config:
        try:
            config = load_config(parsed_args.config)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1

    ticker = parsed_args.ticker

    if parsed_args.sentiment or parsed_args.sentiment_vader:
        try:
            from .sentiment import SentimentAnalyzer, format_sentiment_report

            use_vader = parsed_args.sentiment_vader
            analyzer = SentimentAnalyzer(use_vader=use_vader)

            if not parsed_args.quiet:
                print(f"\nAnalyzing news sentiment for {ticker}...")
                if not use_vader:
                    print("   Loading FinBERT model (first run may take a while)...")

            result = analyzer.analyze(ticker, max_articles=20)

            if parsed_args.export:
                import json

                filepath = parsed_args.output or f"{ticker}_sentiment.json"
                with open(filepath, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"Exported to: {filepath}")
            else:
                print(format_sentiment_report(result))

            return 0

        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(
                "\nTo use sentiment analysis, install required packages:",
                file=sys.stderr,
            )
            if parsed_args.sentiment_vader:
                print("   pip install vaderSentiment", file=sys.stderr)
            else:
                print("   pip install transformers torch", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Sentiment analysis failed: {e}", file=sys.stderr)
            return 1

    try:
        analyzer = IDXAnalyzer(ticker, config=config)

        period = parsed_args.period
        if period is None and config:
            period = config.analysis.default_period
        elif period is None:
            period = "6mo"

        if not parsed_args.quiet:
            print(f"\nAnalyzing {analyzer.ticker}...")
            print(f"   Fetching {period} of historical data...")

        analyzer.fetch_data(period=period)
        result = analyzer.analyze()

        if parsed_args.chart:
            if not parsed_args.quiet:
                print(f"   Generating chart...")
            chart_path = analyzer.generate_chart(
                output_path=parsed_args.chart_output, show=False
            )
            if not parsed_args.quiet:
                print(f"Chart saved: {chart_path}")

        if parsed_args.export:
            if parsed_args.output:
                filepath = parsed_args.output
            else:
                filepath = f"{result.ticker}_analysis.{parsed_args.export}"

            if parsed_args.export == "csv":
                export_to_csv(result, filepath)
            else:
                export_to_json(result, filepath)

        if parsed_args.chat:
            print(analyzer.generate_chat_report(result))
            return 0

        if not parsed_args.quiet:
            try:
                from rich.console import Console

                console = Console()
                report = analyzer.generate_rich_report(result)
                console.print(report)
            except ImportError:
                print(format_output(result, quiet=False))
        else:
            print(format_output(result, quiet=True))

        return 0

    except IDXAnalyzerError as e:
        print(format_error_for_user(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
