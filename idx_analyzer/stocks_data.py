"""
IDX Stock Data Loader

Loads stock universe from JSON file for better manageability.
Structured similar to indonesian_sentiment_lexicon.json with _meta section
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _get_data_path() -> Path:
    """Get path to data directory."""
    return Path(__file__).parent / "data"


def _load_json(filename: str) -> dict:
    """Load JSON file from data directory."""
    filepath = _get_data_path() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Load stock data
_stocks_data = None


def _get_stocks_data() -> dict:
    """Lazy load stocks data."""
    global _stocks_data
    if _stocks_data is None:
        _stocks_data = _load_json("idx_stocks.json")
    return _stocks_data


# ============================================================================
# INDEX DATA
# ============================================================================


def get_lq45_tickers() -> List[str]:
    """Get LQ45 index constituents (45 most liquid stocks)."""
    data = _get_stocks_data()
    return data["indices"]["LQ45"]["tickers"]


def get_idx30_tickers() -> List[str]:
    """Get IDX30 index constituents (30 largest and most liquid)."""
    data = _get_stocks_data()
    return data["indices"]["IDX30"]["tickers"]


def get_jii_tickers() -> List[str]:
    """Get JII (Jakarta Islamic Index) constituents."""
    data = _get_stocks_data()
    return data["indices"]["JII"]["tickers"]


# ============================================================================
# SECTOR DATA
# ============================================================================


def get_sector_tickers(sector: str) -> List[str]:
    """Get all tickers in a given sector."""
    data = _get_stocks_data()
    sector_key = sector.lower().replace(" ", "_")

    if sector_key in data["sectors"]:
        return data["sectors"][sector_key]["tickers"]
    return []


def get_tickers_by_sector(sector: str) -> List[str]:
    """Alias for get_sector_tickers - for compatibility."""
    return get_sector_tickers(sector)


def get_all_sectors() -> Dict[str, dict]:
    """Get all sectors data."""
    data = _get_stocks_data()
    return data["sectors"]


def get_sector_for_ticker(ticker: str) -> Optional[str]:
    """Find which sector a ticker belongs to."""
    data = _get_stocks_data()
    clean_ticker = ticker.upper().replace(".JK", "").replace("IDX:", "")

    for sector_key, sector_data in data["sectors"].items():
        if clean_ticker in sector_data["tickers"]:
            return sector_key
    return None


def get_sector_peers(ticker: str, max_peers: int = 4) -> List[str]:
    """Get peer tickers in the same sector (excluding the given ticker)."""
    clean_ticker = ticker.upper().replace(".JK", "").replace("IDX:", "")
    sector = get_sector_for_ticker(clean_ticker)

    if not sector:
        return []

    peers = [t for t in get_sector_tickers(sector) if t != clean_ticker]
    return peers[:max_peers]


# ============================================================================
# BOARD DATA
# ============================================================================


def get_board_info(board_type: str) -> dict:
    """Get information about a specific board."""
    data = _get_stocks_data()
    return data["boards"].get(board_type, {})


def get_board_tickers(board_type: str) -> List[str]:
    """Get tickers for a specific board."""
    data = _get_stocks_data()
    board = data["boards"].get(board_type, {})
    return board.get("tickers", [])


def get_all_boards() -> Dict[str, dict]:
    """Get all boards information."""
    data = _get_stocks_data()
    return data["boards"]


# ============================================================================
# SCREENER DEFAULTS
# ============================================================================


def get_screener_default_universe() -> List[str]:
    """Get default screener universe (LQ45)."""
    return get_lq45_tickers()


def get_expanded_screener_universe() -> List[str]:
    """Get expanded screener universe (IDX30 + LQ45 extras)."""
    lq45 = set(get_lq45_tickers())
    idx30 = set(get_idx30_tickers())

    # Combine both indices
    combined = lq45.union(idx30)
    return sorted(list(combined))


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================


# Keep these for backward compatibility with existing code
def get_all_utama_tickers() -> List[str]:
    """Get all tickers on Utama (Main) board - from JSON."""
    return get_board_tickers("utama")


def get_liquid_tickers() -> List[str]:
    """Get liquid tickers - alias for LQ45."""
    return get_lq45_tickers()


# Legacy sector mappings for backward compatibility
IDX_SECTORS = {
    "Banking": {
        "tickers": get_sector_tickers("banking"),
        "name": "Banking",
        "index": "JII",
    },
    "Telecommunication": {
        "tickers": get_sector_tickers("telecommunication"),
        "name": "Telecommunication",
        "index": "JII",
    },
    "Consumer Goods": {
        "tickers": get_sector_tickers("consumer_goods"),
        "name": "Consumer Goods",
        "index": "JII",
    },
    "Mining": {
        "tickers": get_sector_tickers("mining"),
        "name": "Mining",
        "index": "JII",
    },
    "Technology": {
        "tickers": get_sector_tickers("technology"),
        "name": "Technology",
        "index": "JII",
    },
    "Property": {
        "tickers": get_sector_tickers("property"),
        "name": "Property",
        "index": "JII",
    },
    "Infrastructure": {
        "tickers": get_sector_tickers("infrastructure"),
        "name": "Infrastructure",
        "index": "JII",
    },
    "Energy": {
        "tickers": get_sector_tickers("energy"),
        "name": "Energy",
        "index": "JII",
    },
    "Healthcare": {
        "tickers": get_sector_tickers("healthcare"),
        "name": "Healthcare",
        "index": "JII",
    },
    "Finance": {
        "tickers": get_sector_tickers("finance"),
        "name": "Finance",
        "index": "JII",
    },
    "Automotive": {
        "tickers": get_sector_tickers("automotive"),
        "name": "Automotive",
        "index": "JII",
    },
    "Retail": {
        "tickers": get_sector_tickers("retail"),
        "name": "Retail",
        "index": "JII",
    },
    "Cement": {
        "tickers": get_sector_tickers("cement"),
        "name": "Cement",
        "index": "JII",
    },
    "Tobacco": {
        "tickers": get_sector_tickers("tobacco"),
        "name": "Tobacco",
        "index": "JII",
    },
    "Palm Oil": {
        "tickers": get_sector_tickers("palm_oil"),
        "name": "Palm Oil",
        "index": "JII",
    },
    "Towers": {
        "tickers": get_sector_tickers("towers"),
        "name": "Towers",
        "index": "JII",
    },
    "Coal": {
        "tickers": get_sector_tickers("coal"),
        "name": "Coal",
        "index": "JII",
    },
    "Nickel": {
        "tickers": get_sector_tickers("nickel"),
        "name": "Nickel",
        "index": "JII",
    },
    "Logistics": {
        "tickers": get_sector_tickers("logistics"),
        "name": "Logistics",
        "index": "JII",
    },
    "Construction": {
        "tickers": get_sector_tickers("construction"),
        "name": "Construction",
        "index": "JII",
    },
    "Toll Roads": {
        "tickers": get_sector_tickers("toll_roads"),
        "name": "Toll Roads",
        "index": "JII",
    },
    "Gas": {
        "tickers": get_sector_tickers("gas"),
        "name": "Gas",
        "index": "JII",
    },
}

TICKER_TO_SECTOR = {}
for sector_name, sector_data in IDX_SECTORS.items():
    for ticker in sector_data["tickers"]:
        TICKER_TO_SECTOR[ticker] = sector_name


def get_all_tickers_in_sector(sector: str) -> List[str]:
    """Backward compatible function."""
    return get_sector_tickers(sector)
