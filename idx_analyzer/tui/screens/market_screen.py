"""
Market Overview Screen for IDX Terminal
Displays IHSG, LQ45, IDX30, top gainers/losers, and sector overview
"""

import logging

import asyncio
from datetime import datetime

import yfinance as yf
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Label, LoadingIndicator, Static

from ...stocks_data import get_liquid_tickers, get_lq45_tickers

# Top liquid stocks for market overview
LIQUID_STOCKS = [t + ".JK" for t in get_liquid_tickers()]
from ..themes import COLORS
from ..widgets.market_table import MarketTable
from ..widgets.price_display import PriceDisplay

logger = logging.getLogger(__name__)

class MarketScreen(Screen):
    """Market overview screen with indices and movers"""

    CSS = """
    #market-container {
        layout: grid;
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: auto 1fr 1fr;
        height: 100%;
        padding: 1;
    }

    #indices-panel {
        column-span: 3;
        height: auto;
        layout: horizontal;
    }

    .index-card {
        width: 1fr;
        height: auto;
        border: solid $surface;
        padding: 1;
        margin: 0 1;
    }

    .index-name {
        text-style: bold;
        color: $primary;
    }

    #movers-panel {
        row-span: 2;
        border: solid $surface;
        padding: 1;
    }

    #gainers-table, #losers-table {
        height: 1fr;
        border: none;
    }

    #watchlist-panel, #sectors-panel {
        border: solid $surface;
        padding: 1;
    }

    .panel-title {
        color: $primary;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    .loading {
        align: center middle;
        color: $primary;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "chart", "Chart"),
        ("enter", "select", "Select"),
    ]

    # Reactive data
    ihsg_data: reactive[dict] = reactive(dict)
    lq45_data: reactive[dict] = reactive(dict)
    idx30_data: reactive[dict] = reactive(dict)
    top_gainers: reactive[list] = reactive(list)
    top_losers: reactive[list] = reactive(list)
    last_update: reactive[str] = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loading = False

    def compose(self) -> ComposeResult:
        """Compose market screen layout"""
        with Vertical(id="market-container"):
            # Indices Panel (top row)
            with Horizontal(id="indices-panel"):
                # IHSG
                with Vertical(classes="index-card"):
                    yield Label("📊 IHSG (JKSE)", classes="index-name")
                    yield PriceDisplay(id="ihsg-price")

                # LQ45
                with Vertical(classes="index-card"):
                    yield Label("📈 LQ45", classes="index-name")
                    yield PriceDisplay(id="lq45-price")

                # IDX30
                with Vertical(classes="index-card"):
                    yield Label("📉 IDX30", classes="index-name")
                    yield PriceDisplay(id="idx30-price")

            # Top Gainers
            with Vertical(id="movers-panel"):
                yield Label("🚀 TOP GAINERS", classes="panel-title")
                yield MarketTable(id="gainers-table")

            # Top Losers
            with Vertical(id="watchlist-panel"):
                yield Label("📉 TOP LOSERS", classes="panel-title")
                yield MarketTable(id="losers-table")

            # Sector Overview
            with Vertical(id="sectors-panel"):
                yield Label("🏭 SECTOR OVERVIEW", classes="panel-title")
                yield Static("Sector data coming soon...", id="sectors-content")

            # Footer hint
            yield Static(
                "[dim]↑↓=Navigate | Enter=Profile | C=Chart | R=Refresh[/dim]",
                id="footer-hint",
            )

    def on_mount(self) -> None:
        """Initialize and load data"""
        # Setup tables
        gainers = self.query_one("#gainers-table", MarketTable)
        gainers.add_columns("Ticker", "Name", "Price", "Change", "%", "Volume")
        gainers.focus()

        losers = self.query_one("#losers-table", MarketTable)
        losers.add_columns("Ticker", "Name", "Price", "Change", "%", "Volume")

        # Load data (async method needs to be called properly)
        self.run_worker(self.refresh_data())

        # Set up auto-refresh every 60 seconds
        self.set_interval(60, self.refresh_data)

    async def refresh_data(self) -> None:
        """Refresh market data"""
        if self._loading:
            return

        self._loading = True

        try:
            # Fetch index data
            await self._fetch_indices()

            # Fetch movers
            await self._fetch_movers()

            # Update timestamp
            self.last_update = datetime.now().strftime("%H:%M:%S")

        except Exception as e:
            self.app.notify(f"Error loading market data: {e}", severity="error")
        finally:
            self._loading = False

    async def _fetch_indices(self) -> None:
        """Fetch index data from Yahoo Finance"""
        try:
            # IHSG
            ihsg = yf.Ticker("^JKSE")
            ihsg_hist = ihsg.history(period="2d")
            if len(ihsg_hist) >= 2:
                current = ihsg_hist["Close"].iloc[-1]
                prev = ihsg_hist["Close"].iloc[-2]
                change = current - prev
                change_pct = (change / prev) * 100

                price_display = self.query_one("#ihsg-price", PriceDisplay)
                price_display.set_price(current, change, change_pct)

            # LQ45 (using FQLQ45.JK or approximate with ^JKLQ45)
            try:
                lq45 = yf.Ticker("^JKLQ45")
                lq45_hist = lq45.history(period="2d")
                if len(lq45_hist) >= 2:
                    current = lq45_hist["Close"].iloc[-1]
                    prev = lq45_hist["Close"].iloc[-2]
                    change = current - prev
                    change_pct = (change / prev) * 100

                    price_display = self.query_one("#lq45-price", PriceDisplay)
                    price_display.set_price(current, change, change_pct)
            except Exception as e:
                logger.debug(f"LQ45 fetch error: {e}")

            # IDX30
            try:
                idx30 = yf.Ticker("^JKIDX30")
                idx30_hist = idx30.history(period="2d")
                if len(idx30_hist) >= 2:
                    current = idx30_hist["Close"].iloc[-1]
                    prev = idx30_hist["Close"].iloc[-2]
                    change = current - prev
                    change_pct = (change / prev) * 100

                    price_display = self.query_one("#idx30-price", PriceDisplay)
                    price_display.set_price(current, change, change_pct)
            except Exception as e:
                logger.debug(f"IDX30 fetch error: {e}")

        except Exception as e:
            self.app.notify(f"Index fetch error: {e}", severity="warning")

    async def _fetch_movers(self) -> None:
        """Fetch top gainers and losers from liquid stocks"""
        movers = []

        # Sample top liquid stocks
        sample_tickers = LIQUID_STOCKS[:30]  # Top 30 liquid

        try:
            # Batch download
            data = yf.download(
                sample_tickers, period="2d", progress=False, auto_adjust=False
            )

            if data.empty:
                return

            # Calculate changes
            for ticker in sample_tickers:
                try:
                    if len(data["Close"].columns) > 1:
                        # Multiple tickers
                        if ticker not in data["Close"].columns:
                            continue
                        closes = data["Close"][ticker]
                        volumes = data["Volume"][ticker]
                    else:
                        # Single ticker
                        closes = data["Close"]
                        volumes = data["Volume"]

                    if len(closes) < 2:
                        continue

                    current = closes.iloc[-1]
                    prev = closes.iloc[-2]
                    change = current - prev
                    change_pct = (change / prev) * 100
                    volume = volumes.iloc[-1]

                    movers.append(
                        {
                            "ticker": ticker.replace(".JK", ""),
                            "price": current,
                            "change": change,
                            "change_pct": change_pct,
                            "volume": volume,
                        }
                    )
                except Exception:
                    continue

            # Sort by change
            movers.sort(key=lambda x: x["change_pct"], reverse=True)

            # Update tables
            gainers = self.query_one("#gainers-table", MarketTable)
            losers = self.query_one("#losers-table", MarketTable)

            gainers.clear()
            losers.clear()

            # Top 5 gainers (positive change only)
            for m in movers[:5]:
                if m["change_pct"] > 0:
                    gainers.add_price_row(
                        m["ticker"],
                        "",  # Name lookup can be added later
                        m["price"],
                        m["change"],
                        m["change_pct"],
                        m["volume"],
                    )

            # Top 5 losers (negative change only)
            for m in movers[-5:]:
                if m["change_pct"] < 0:
                    losers.add_price_row(
                        m["ticker"],
                        "",
                        m["price"],
                        m["change"],
                        m["change_pct"],
                        m["volume"],
                    )

        except Exception as e:
            self.app.notify(f"Movers fetch error: {e}", severity="warning")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (Enter/Click) - navigate to stock"""
        try:
            ticker = event.row_key.value
            if ticker:
                self.app.show_stock(ticker)
        except Exception as e:
            self.app.notify(f"Error selecting row: {e}", severity="error")

    def action_debug(self) -> None:
        """Debug - show current state"""
        for table_id in ["#gainers-table", "#losers-table"]:
            table = self.query_one(table_id, MarketTable)
            self.app.notify(
                f"{table_id}: cursor={table.cursor_row}, rows={table.row_count}",
                severity="information",
            )

    def action_refresh(self) -> None:
        """Refresh market data"""
        self.run_worker(self.refresh_data())
        self.app.notify("Refreshing...", severity="information")

    def _get_selected_ticker(self) -> str | None:
        """Get ticker from whichever table has cursor focus"""
        for table_id in ["#gainers-table", "#losers-table"]:
            table = self.query_one(table_id, MarketTable)
            if table.cursor_row is not None and table.cursor_row >= 0:
                try:
                    row = table.get_row_at(table.cursor_row)
                    if row and len(row) > 0:
                        return str(row[0]).strip()
                except Exception as e:
                    self.app.notify(f"Error getting row: {e}", severity="error")
        return None

    def action_select(self) -> None:
        """Handle enter key on table - open stock profile"""
        ticker = self._get_selected_ticker()
        if ticker:
            self.app.show_stock(ticker)
        else:
            self.app.notify("Select a ticker first (use ↑↓ arrows)", severity="warning")

    def action_chart(self) -> None:
        """Open chart for selected/cursor ticker"""
        ticker = self._get_selected_ticker()
        if ticker:
            self.app.show_stock_chart(ticker)
        else:
            self.app.notify("Select a ticker first (use ↑↓ arrows)", severity="warning")
