"""
Command-line interface for IDX Analyzer
"""

import argparse
import sys
from typing import Optional
from .analyzer import IDXAnalyzer


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="idx-analyzer",
        description="""
╔══════════════════════════════════════════════════════════════╗
║              IDX Stock Analyzer v1.0.0                       ║
║     Indonesian Stock Market Technical Analysis Tool          ║
╚══════════════════════════════════════════════════════════════╝

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
        default="6mo",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        help="Historical data period (default: 6mo)",
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

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    return parser


def format_output(result, quiet: bool = False) -> str:
    """Format analysis output"""
    if quiet:
        # Minimal output for scripting
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

    # Full formatted output
    output = []
    output.append("")
    output.append("╔" + "═" * 62 + "╗")
    output.append("║" + f" IDX Stock Analysis: {result.ticker:^42} " + "║")
    output.append("╚" + "═" * 62 + "╝")
    output.append("")

    # Price info
    change_color = "🟢" if result.change_percent >= 0 else "🔴"
    output.append(f"💰 Current Price: {result.current_price:>12,.0f} IDR")
    output.append(f"   Daily Change:  {change_color} {result.change_percent:>+10.2f}%")
    output.append(f"   Volume:        {result.volume:>12,}")
    output.append("")

    # 52-week range
    output.append("━" * 64)
    output.append("📊 52-WEEK RANGE")
    output.append("━" * 64)
    from_high = (result.current_price - result.week_52_high) / result.week_52_high * 100
    from_low = (result.current_price - result.week_52_low) / result.week_52_low * 100
    output.append(
        f"   High: {result.week_52_high:>10,.0f}  ({from_high:+.1f}% from current)"
    )
    output.append(
        f"   Low:  {result.week_52_low:>10,.0f}  (+{from_low:.1f}% from current)"
    )
    output.append("")

    # Support levels
    output.append("━" * 64)
    output.append("🟢 SUPPORT LEVELS (Buy Zones)")
    output.append("━" * 64)
    if result.support_levels:
        for i, s in enumerate(result.support_levels[:4], 1):
            dist = (result.current_price - s.level) / result.current_price * 100
            strength_icon = (
                "⭐"
                if s.strength == "strong"
                else "▪"
                if s.strength == "moderate"
                else "•"
            )
            output.append(
                f"   {i}. {s.level:>8,.0f}  ({dist:>5.1f}% below)  {strength_icon} {s.strength}"
            )
    else:
        output.append("   No clear support levels identified")
    output.append("")

    # Resistance levels
    output.append("━" * 64)
    output.append("🔴 RESISTANCE LEVELS (Sell/Target Zones)")
    output.append("━" * 64)
    if result.resistance_levels:
        for i, r in enumerate(result.resistance_levels[:4], 1):
            dist = (r.level - result.current_price) / result.current_price * 100
            strength_icon = (
                "⭐"
                if r.strength == "strong"
                else "▪"
                if r.strength == "moderate"
                else "•"
            )
            output.append(
                f"   {i}. {r.level:>8,.0f}  (+{dist:>5.1f}% above)  {strength_icon} {r.strength}"
            )
    else:
        output.append("   No clear resistance levels identified")
    output.append("")

    output.append("━" * 64)
    output.append("📊 MOVING AVERAGES")
    output.append("━" * 64)

    sma_20_pct = (result.current_price - result.sma_20) / result.sma_20 * 100
    sma_50_pct = (result.current_price - result.sma_50) / result.sma_50 * 100

    icon_20 = "🟢" if result.current_price > result.sma_20 else "🔴"
    icon_50 = "🟢" if result.current_price > result.sma_50 else "🔴"

    output.append(f"   SMA 20: {result.sma_20:>10,.0f}  {icon_20} {sma_20_pct:+.1f}%")
    output.append(f"   SMA 50: {result.sma_50:>10,.0f}  {icon_50} {sma_50_pct:+.1f}%")

    if result.sma_200:
        sma_200_pct = (result.current_price - result.sma_200) / result.sma_200 * 100
        icon_200 = "🟢" if result.current_price > result.sma_200 else "🔴"
        output.append(
            f"   SMA 200: {result.sma_200:>9,.0f}  {icon_200} {sma_200_pct:+.1f}%"
        )

    output.append("")

    # Trend and recommendation
    output.append("━" * 64)
    trend_icon = (
        "🐂" if "Bull" in result.trend else "🐻" if "Bear" in result.trend else "⚖"
    )
    output.append(f"📈 Trend: {trend_icon} {result.trend}")
    output.append("")
    output.append("💡 RECOMMENDATION:")
    output.append(f"   {result.recommendation}")
    output.append("")
    output.append("━" * 64)
    output.append("")

    return "\n".join(output)


def export_to_csv(result, filepath: str):
    """Export analysis to CSV"""
    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Level", "Distance %", "Strength", "Description"])

        # Current price
        writer.writerow(["Current", result.current_price, "", "", ""])

        # Supports
        for s in result.support_levels:
            dist = (result.current_price - s.level) / result.current_price * 100
            writer.writerow(
                ["Support", s.level, f"{dist:.2f}", s.strength, s.description]
            )

        # Resistances
        for r in result.resistance_levels:
            dist = (r.level - result.current_price) / result.current_price * 100
            writer.writerow(
                ["Resistance", r.level, f"{dist:.2f}", r.strength, r.description]
            )

    print(f"✅ Exported to: {filepath}")


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
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Exported to: {filepath}")


def main(args: Optional[list] = None) -> int:
    """Main entry point"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Format ticker
    ticker = parsed_args.ticker

    try:
        # Create analyzer
        analyzer = IDXAnalyzer(ticker)

        if not parsed_args.quiet:
            print(f"\n🔍 Analyzing {analyzer.ticker}...")
            print(f"   Fetching {parsed_args.period} of historical data...")

        # Fetch data
        if not analyzer.fetch_data(period=parsed_args.period):
            print(f"❌ Error: Could not fetch data for {ticker}", file=sys.stderr)
            return 1

        # Analyze
        result = analyzer.analyze()

        # Generate chart if requested
        if parsed_args.chart:
            if not parsed_args.quiet:
                print(f"   Generating chart...")
            chart_path = analyzer.generate_chart(
                output_path=parsed_args.chart_output, show=False
            )
            if not parsed_args.quiet:
                print(f"✅ Chart saved: {chart_path}")

        # Export if requested
        if parsed_args.export:
            if parsed_args.output:
                filepath = parsed_args.output
            else:
                filepath = f"{result.ticker}_analysis.{parsed_args.export}"

            if parsed_args.export == "csv":
                export_to_csv(result, filepath)
            else:
                export_to_json(result, filepath)

        # Print output
        print(format_output(result, quiet=parsed_args.quiet))

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
