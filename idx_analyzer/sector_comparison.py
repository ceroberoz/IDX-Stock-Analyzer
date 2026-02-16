"""
Sector comparison module for comparing stocks within the same sector.
Fetches peer data from Yahoo Finance and formats for display.
"""

from dataclasses import dataclass
from typing import Optional

import yfinance as yf

# Ensure cache is configured (idempotent - safe to call multiple times)
from .cache import configure_yfinance_cache
from .sectors_data import get_sector_for_ticker, get_sector_peers

configure_yfinance_cache()


@dataclass
class PeerData:
    """Data for a single peer stock."""

    ticker: str
    price: float
    change_percent: float
    volume: int
    is_up: bool


def fetch_peer_data(ticker: str) -> Optional[PeerData]:
    """
    Fetch current price and change % for a single ticker.
    Uses minimal data fetch for speed.
    """
    try:
        stock = yf.Ticker(f"{ticker}.JK")
        hist = stock.history(period="2d", interval="1d")

        if len(hist) < 2:
            return None

        current_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2])
        change_pct = (current_price - prev_price) / prev_price * 100
        volume = int(hist["Volume"].iloc[-1])

        return PeerData(
            ticker=ticker,
            price=current_price,
            change_percent=change_pct,
            volume=volume,
            is_up=change_pct >= 0,
        )
    except Exception:
        return None


def get_sector_comparison(
    ticker: str,
    max_peers: int = 4,
    include_ticker: bool = False,
) -> tuple[str, list[PeerData]]:
    """
    Get sector comparison data for a ticker.

    Returns:
        Tuple of (sector_name, list of PeerData)
    """
    clean_ticker = ticker.upper().replace(".JK", "").replace("IDX:", "")
    sector = get_sector_for_ticker(clean_ticker)

    if not sector:
        return "Unknown", []

    # Get peer tickers
    peer_tickers = get_sector_peers(clean_ticker, max_peers=max_peers)

    # Include the original ticker if requested
    if include_ticker:
        peer_tickers = [clean_ticker] + peer_tickers
        peer_tickers = peer_tickers[:max_peers]

    # Fetch data for each peer
    peers_data = []
    for peer in peer_tickers:
        data = fetch_peer_data(peer)
        if data:
            peers_data.append(data)

    # Sort by change % (descending)
    peers_data.sort(key=lambda x: x.change_percent, reverse=True)

    return sector, peers_data


def format_peer_line(peer: PeerData, compact: bool = False) -> str:
    """
    Format a single peer data into user-friendly string.

    Format: (UP 8250 BBCA) or (DOWN 6800 BMRI)
    """
    direction = "UP" if peer.is_up else "DOWN"
    arrow = "🟢" if peer.is_up else "🔴"

    if compact:
        return f"{arrow} {direction} {peer.price:,.0f} {peer.ticker}"
    else:
        change_str = f"{peer.change_percent:+.2f}%"
        return (
            f"{arrow} {direction} {peer.price:>8,.0f} {peer.ticker:<6} {change_str:>8}"
        )


def format_sector_comparison(
    ticker: str,
    max_peers: int = 4,
    compact: bool = False,
) -> str:
    """
    Format sector comparison as a readable string.

    Example output:
    📊 Sector Peers (Banking):
    🟢 UP   8,250 BBCA   +1.50%
    🟢 UP   5,100 BBRI   +0.80%
    🔴 DOWN 6,800 BMRI   -0.50%
    """
    sector, peers = get_sector_comparison(ticker, max_peers=max_peers)

    if not peers:
        return f"📊 No sector data available for {ticker}"

    lines = [f"📊 Sector Peers ({sector}):"]

    for peer in peers:
        lines.append(f"  {format_peer_line(peer, compact=compact)}")

    return "\n".join(lines)


def format_peer_ticker_style(ticker: str) -> str:
    """
    Format peer data in the user's requested style: (UP 1000 CDIA)

    Returns a single line string like:
    (🟢 UP 1000 CDIA)  (🔴 DOWN 850 ADRO)
    """
    sector, peers = get_sector_comparison(ticker, max_peers=3, include_ticker=True)

    if not peers:
        return ""

    segments = []
    for peer in peers:
        direction = "UP" if peer.is_up else "DOWN"
        arrow = "🟢" if peer.is_up else "🔴"
        segments.append(f"({arrow} {direction} {peer.price:,.0f} {peer.ticker})")

    return "  ".join(segments)


def get_peer_table_data(
    ticker: str, max_peers: int = 4, include_ticker: bool = True
) -> dict:
    """
    Get structured data for table/chart display.

    Returns dict with:
    - sector: str
    - peers: list of PeerData
    - up_count: int
    - down_count: int
    """
    sector, peers = get_sector_comparison(
        ticker, max_peers=max_peers, include_ticker=include_ticker
    )

    up_count = sum(1 for p in peers if p.is_up)
    down_count = len(peers) - up_count

    return {
        "sector": sector,
        "peers": peers,
        "up_count": up_count,
        "down_count": down_count,
        "total": len(peers),
    }
