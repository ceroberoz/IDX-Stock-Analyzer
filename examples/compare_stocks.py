#!/usr/bin/env python3
"""
Compare Multiple Stocks Example

This example shows how to compare multiple stocks side-by-side
to find the best opportunities in a sector.
"""

from idx_analyzer.analyzer import IDXAnalyzer
from typing import List, Dict


def analyze_stocks(tickers: List[str]) -> List[Dict]:
    """Analyze multiple stocks and return key metrics."""
    results = []

    for ticker in tickers:
        try:
            analyzer = IDXAnalyzer(ticker)
            if analyzer.fetch_data(period="6mo"):
                result = analyzer.analyze()

                # Calculate additional metrics
                nearest_support = (
                    max([s.level for s in result.support_levels])
                    if result.support_levels
                    else result.current_price * 0.95
                )
                nearest_resistance = (
                    min([r.level for r in result.resistance_levels])
                    if result.resistance_levels
                    else result.current_price * 1.05
                )

                upside = (
                    (nearest_resistance - result.current_price)
                    / result.current_price
                    * 100
                )
                downside = (
                    (result.current_price - nearest_support)
                    / result.current_price
                    * 100
                )
                risk_reward = upside / downside if downside > 0 else 0

                results.append(
                    {
                        "ticker": ticker,
                        "price": result.current_price,
                        "trend": result.trend,
                        "rsi": result.rsi,
                        "upside": upside,
                        "downside": downside,
                        "rr_ratio": risk_reward,
                        "signal": "BUY"
                        if "Bull" in result.trend and result.rsi < 70
                        else "SELL"
                        if "Bear" in result.trend and result.rsi > 30
                        else "HOLD",
                    }
                )
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    return results


def print_comparison(results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print(
        f"{'Ticker':<10} {'Price':>12} {'Trend':<25} {'RSI':>6} {'Upside':>8} {'Downside':>8} {'R/R':>6} {'Signal':<8}"
    )
    print("=" * 100)

    # Sort by R/R ratio (best opportunities first)
    sorted_results = sorted(results, key=lambda x: x["rr_ratio"], reverse=True)

    for r in sorted_results:
        print(
            f"{r['ticker']:<10} {r['price']:>12,.0f} {r['trend']:<25} {r['rsi']:>6.1f} "
            f"{r['upside']:>7.1f}% {r['downside']:>7.1f}% {r['rr_ratio']:>6.1f} {r['signal']:<8}"
        )

    print("=" * 100)

    # Print best opportunities
    buy_signals = [r for r in results if r["signal"] == "BUY"]
    if buy_signals:
        print("\n🟢 BUY SIGNALS:")
        for r in sorted(buy_signals, key=lambda x: x["rr_ratio"], reverse=True)[:3]:
            print(
                f"   {r['ticker']}: R/R = 1:{r['rr_ratio']:.1f}, RSI = {r['rsi']:.1f}"
            )


def main():
    # Banking sector comparison
    print("Analyzing Banking Sector...")
    banking_stocks = ["BBCA", "BBRI", "BMRI", "BBNI", "BRIS"]
    results = analyze_stocks(banking_stocks)

    print("\n📊 BANKING SECTOR COMPARISON")
    print_comparison(results)

    # Top picks
    print("\n🏆 TOP PICKS:")
    best_rr = max(results, key=lambda x: x["rr_ratio"])
    print(f"   Best Risk/Reward: {best_rr['ticker']} (1:{best_rr['rr_ratio']:.1f})")

    lowest_rsi = min(results, key=lambda x: x["rsi"])
    if lowest_rsi["rsi"] < 40:
        print(
            f"   Potential Bounce: {lowest_rsi['ticker']} (RSI: {lowest_rsi['rsi']:.1f})"
        )


if __name__ == "__main__":
    main()
