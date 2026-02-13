#!/usr/bin/env python3
"""
Basic Usage Example for IDX Stock Analyzer

This example shows how to use the IDXAnalyzer programmatically
from Python code instead of the CLI.
"""

from idx_analyzer.analyzer import IDXAnalyzer


def main():
    ticker = "BBCA"
    print(f"Analyzing {ticker}...\n")

    analyzer = IDXAnalyzer(ticker)

    success = analyzer.fetch_data(period="1y")

    if not success:
        print(f"Failed to fetch data for {ticker}")
        return

    result = analyzer.analyze()

    print(f"Ticker: {result.ticker}")
    print(f"Current Price: {result.current_price:,.0f} IDR")
    print(f"Daily Change: {result.change_percent:+.2f}%")
    print(f"RSI: {result.rsi:.1f}")
    print(f"Trend: {result.trend}")
    print()

    print("Moving Averages:")
    print(f"  SMA 20: {result.sma_20:,.0f}")
    print(f"  SMA 50: {result.sma_50:,.0f}")
    if result.sma_200:
        print(f"  SMA 200: {result.sma_200:,.0f}")
    print()

    print("Support Levels:")
    for i, support in enumerate(result.support_levels[:3], 1):
        dist = (result.current_price - support.level) / result.current_price * 100
        print(f"  {i}. {support.level:,.0f} ({dist:.1f}% below) - {support.strength}")
    print()

    print("Resistance Levels:")
    for i, resistance in enumerate(result.resistance_levels[:3], 1):
        dist = (resistance.level - result.current_price) / result.current_price * 100
        print(f"  {i}. {resistance.level:,.0f} (+{dist:.1f}%) - {resistance.strength}")
    print()

    print(f"Recommendation: {result.recommendation}")
    print()

    chart_path = analyzer.generate_chart(output_path=f"{ticker}_example_chart.png")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
