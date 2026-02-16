"""
IDX Sector data and peer mappings for Indonesian stocks.
"""

# Map of sectors to their major tickers on IDX
IDX_SECTORS = {
    "Banking": {
        "tickers": [
            "BBCA",
            "BBRI",
            "BMRI",
            "BBNI",
            "BRIS",
            "BJTM",
            "BBTN",
            "BJBR",
            "BMRI",
            "BNGA",
        ],
        "name": "Banking",
        "index": "JII",
    },
    "Telecommunication": {
        "tickers": ["TLKM", "ISAT", "EXCL", "FREN", "TBIG", "TOWR", "GOLD"],
        "name": "Telecommunication",
        "index": "JII",
    },
    "Consumer Goods": {
        "tickers": [
            "UNVR",
            "ICBP",
            "MYOR",
            "GGRM",
            "KAEF",
            "INDF",
            "SIMP",
            "AALI",
            "LSIP",
            "SSMS",
        ],
        "name": "Consumer Goods",
        "index": "JII",
    },
    "Mining": {
        "tickers": [
            "ADRO",
            "ITMG",
            "PTBA",
            "ANTM",
            "INCO",
            "TINS",
            "HRUM",
            "DOID",
            "KKGI",
            "MBAP",
        ],
        "name": "Mining",
        "index": "JII",
    },
    "Technology": {
        "tickers": [
            "GOTO",
            "BELI",
            "BUKA",
            "MCAS",
            "KREN",
            "INDO",
            "DIVA",
            "ENVY",
            "DMMX",
            "NFCX",
        ],
        "name": "Technology",
        "index": "JII",
    },
    "Property": {
        "tickers": [
            "CTRA",
            "SMRA",
            "PWON",
            "BSDE",
            "LPKR",
            "APLN",
            "MEGA",
            "TOTL",
            "DILD",
            "JRPT",
        ],
        "name": "Property",
        "index": "JII",
    },
    "Infrastructure": {
        "tickers": [
            "ASII",
            "ICBP",
            "JPFA",
            "MAIN",
            "SMSM",
            "CMRY",
            "GOOD",
            "AVIA",
            "MIBA",
            "PSDN",
        ],
        "name": "Infrastructure",
        "index": "JII",
    },
    "Energy": {
        "tickers": [
            "MEDC",
            "ELSA",
            "ENRG",
            "ESSA",
            "INDO",
            "TPIA",
            "AKRA",
            "BUMI",
            "DOID",
            "TOBA",
        ],
        "name": "Energy",
        "index": "JII",
    },
    "Healthcare": {
        "tickers": [
            "KLBF",
            "SIDO",
            "SCPI",
            "TSPC",
            "DYDX",
            "MEDS",
            "HEAL",
            "OMED",
            "GIAA",
            "PRDA",
        ],
        "name": "Healthcare",
        "index": "JII",
    },
    "Finance": {
        "tickers": [
            "BFIN",
            "ADMF",
            "WOMF",
            "LPPS",
            "BBLD",
            "MAPI",
            "MPPA",
            "ERAA",
            "AUTO",
            "SMSM",
        ],
        "name": "Finance",
        "index": "JII",
    },
}

# Reverse mapping: ticker -> sector
TICKER_TO_SECTOR = {}
for sector_name, sector_data in IDX_SECTORS.items():
    for ticker in sector_data["tickers"]:
        TICKER_TO_SECTOR[ticker] = sector_name


def get_sector_for_ticker(ticker: str) -> str | None:
    """Get sector name for a given ticker."""
    clean_ticker = ticker.upper().replace(".JK", "").replace("IDX:", "")
    return TICKER_TO_SECTOR.get(clean_ticker)


def get_sector_peers(ticker: str, max_peers: int = 4) -> list[str]:
    """Get peer tickers in the same sector (excluding the given ticker)."""
    clean_ticker = ticker.upper().replace(".JK", "").replace("IDX:", "")
    sector = get_sector_for_ticker(clean_ticker)

    if not sector:
        return []

    peers = [t for t in IDX_SECTORS[sector]["tickers"] if t != clean_ticker]

    return peers[:max_peers]


def get_all_tickers_in_sector(sector: str) -> list[str]:
    """Get all tickers in a given sector."""
    sector_key = sector.title()
    if sector_key in IDX_SECTORS:
        return IDX_SECTORS[sector_key]["tickers"]
    return []
