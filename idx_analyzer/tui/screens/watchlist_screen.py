"""
Watchlist Screen for IDX Terminal
User-configurable watchlist with auto-refresh
"""

import logging

from datetime import datetime
from pathlib import Path

import yfinance as yf
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, Static

from ...config import load_config, save_config
from ..widgets.market_table import MarketTable
from ..widgets.sparkline import Sparkline

logger = logging.getLogger(__name__)


class WatchlistScreen(Screen):
    """Watchlist screen with user-configurable stocks"""

    CSS = """
    #watchlist-container {
        layout: vertical;
        height: 100%;
        padding: 1;
    }

    #watchlist-header {
        height: auto;
        layout: horizontal;
    }

    #watchlist-title {
        width: auto;
        color: $primary;
        text-style: bold;
    }

    #last-update {
        width: auto;
        color: $text;
    }

    #add-ticker-row {
        height: auto;
        layout: horizontal;
        margin: 1 0;
    }

    #ticker-input {
        width: 20;
        margin-right: 1;
    }

    #add-btn {
        width: auto;
    }

    #watchlist-table {
        height: 1fr;
        border: solid $surface;
    }

    .panel-title {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    .empty-message {
        align: center middle;
        color: $text;
        text-style: dim;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "chart", "Chart"),
        ("delete", "remove", "Remove"),
        ("enter", "select", "Select"),
    ]

    # Default watchlist
    DEFAULT_WATCHLIST = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "GOTO.JK"]

    # Reactive state
    watchlist: reactive[list] = reactive(list)
    stock_data: reactive[dict] = reactive(dict)
    last_update: reactive[str] = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loading = False
        self._config_file = Path.home() / ".config" / "idx-analyzer" / "watchlist.toml"

    def compose(self) -> ComposeResult:
        """Compose watchlist screen"""
        with Vertical(id="watchlist-container"):
            with Horizontal(id="watchlist-header"):
                yield Label("⭐ MY WATCHLIST", id="watchlist-title")
                yield Static(" | ")
                yield Label("Last update: --:--:--", id="last-update")

            # Add ticker input
            with Horizontal(id="add-ticker-row"):
                yield Input(placeholder="TICKER", id="ticker-input")
                yield Button("+ Add", id="add-btn", variant="primary")

            # Watchlist table
            yield MarketTable(id="watchlist-table")

            yield Static(
                "[dim]Press Enter on ticker to view details | Delete to remove[/dim]",
                id="footer-hint",
            )

    def on_mount(self) -> None:
        """Initialize watchlist"""
        # Setup table
        table = self.query_one("#watchlist-table", MarketTable)
        table.add_columns("Ticker", "Name", "Price", "Change", "%", "Volume", "Trend")
        table.focus()

        # Load watchlist
        self.load_watchlist()

        # Refresh data
        self.run_worker(self.refresh_data())

        # Auto-refresh every 30 seconds
        self.set_interval(30, self.refresh_data)

    def load_watchlist(self) -> None:
        """Load watchlist from config"""
        try:
            if self._config_file.exists():
                import tomli

                with open(self._config_file, "rb") as f:
                    config = tomli.load(f)
                    self.watchlist = config.get("watchlist", self.DEFAULT_WATCHLIST)
            else:
                self.watchlist = self.DEFAULT_WATCHLIST
                self.save_watchlist()
        except Exception:
            self.watchlist = self.DEFAULT_WATCHLIST

    def save_watchlist(self) -> None:
        """Save watchlist to config"""
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            import tomli_w

            with open(self._config_file, "wb") as f:
                tomli_w.dump({"watchlist": self.watchlist}, f)
        except Exception as e:
            self.app.notify(f"Failed to save watchlist: {e}", severity="warning")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle add ticker"""
        if event.input.id == "ticker-input":
            self.add_ticker(event.value.upper().strip())
            event.input.value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id == "add-btn":
            input_widget = self.query_one("#ticker-input", Input)
            ticker = input_widget.value.upper().strip()
            if ticker:
                self.add_ticker(ticker)
                input_widget.value = ""

    def add_ticker(self, ticker: str) -> None:
        """Add ticker to watchlist"""
        # Format ticker
        if not ticker.endswith(".JK"):
            ticker = f"{ticker}.JK"

        if ticker in self.watchlist:
            self.app.notify(f"{ticker} already in watchlist", severity="warning")
            return

        self.watchlist.append(ticker)
        self.save_watchlist()
        self.run_worker(self.refresh_data())
        self.app.notify(f"Added {ticker} to watchlist", severity="information")

    def remove_ticker(self, ticker: str) -> None:
        """Remove ticker from watchlist"""
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
            self.save_watchlist()
            self.run_worker(self.refresh_data())
            self.app.notify(f"Removed {ticker}", severity="information")

    async def refresh_data(self) -> None:
        """Refresh watchlist data"""
        if self._loading or not self.watchlist:
            return

        self._loading = True

        try:
            # Batch download
            data = yf.download(
                self.watchlist, period="5d", progress=False, auto_adjust=False
            )

            if data.empty:
                return

            # Process each ticker
            table = self.query_one("#watchlist-table", MarketTable)
            table.clear()

            for ticker in self.watchlist:
                try:
                    if len(self.watchlist) == 1:
                        # Single ticker
                        closes = data["Close"]
                        volumes = data["Volume"]
                    else:
                        # Multiple tickers
                        if ticker not in data["Close"].columns:
                            continue
                        closes = data["Close"][ticker]
                        volumes = data["Volume"][ticker]

                    if len(closes) < 2:
                        continue

                    current = closes.iloc[-1]
                    prev = closes.iloc[-2]
                    change = current - prev
                    change_pct = (change / prev) * 100
                    volume = volumes.iloc[-1]

                    # Simple trend indicator (5-day)
                    if len(closes) >= 5:
                        trend_5d = closes.iloc[-1] - closes.iloc[-5]
                        trend_icon = (
                            "📈" if trend_5d > 0 else "📉" if trend_5d < 0 else "➡️"
                        )
                    else:
                        trend_icon = "➡️"

                    table.add_price_row(
                        ticker.replace(".JK", ""),
                        "",
                        current,
                        change,
                        change_pct,
                        volume,
                        key=ticker,
                    )

                except Exception:
                    continue

            # Update timestamp
            self.last_update = datetime.now().strftime("%H:%M:%S")
            update_label = self.query_one("#last-update", Label)
            update_label.update(f"Last update: {self.last_update}")

        except Exception as e:
            self.app.notify(f"Watchlist refresh error: {e}", severity="warning")
        finally:
            self._loading = False

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection"""
        ticker = event.row_key.value
        if ticker:
            ticker_clean = ticker.replace(".JK", "")
            self.app.notify(f"Opening {ticker_clean}...", severity="information")
            self.app.show_stock(ticker_clean)

    def action_refresh(self) -> None:
        """Refresh watchlist"""
        self.run_worker(self.refresh_data())
        self.app.notify("Refreshing...", severity="information")

    def _get_selected_ticker(self) -> str | None:
        """Get ticker from selected row"""
        table = self.query_one("#watchlist-table", MarketTable)
        if table.cursor_row is not None and table.cursor_row >= 0:
            try:
                row = table.get_row_at(table.cursor_row)
                if row and len(row) > 0:
                    return str(row[0]).strip()
            except Exception as e:
                self.app.notify(f"Error: {e}", severity="error")
        return None

    def action_remove(self) -> None:
        """Remove selected ticker from watchlist"""
        ticker = self._get_selected_ticker()
        if ticker:
            self.remove_ticker(f"{ticker}.JK")
        else:
            self.app.notify("Select a ticker first", severity="warning")

    def action_select(self) -> None:
        """Handle enter key - open stock profile"""
        ticker = self._get_selected_ticker()
        if ticker:
            self.app.show_stock(ticker.replace(".JK", ""))
        else:
            self.app.notify("Select a ticker first (use ↑↓ arrows)", severity="warning")

    def action_chart(self) -> None:
        """Open chart for cursor ticker"""
        ticker = self._get_selected_ticker()
        if ticker:
            self.app.show_stock_chart(ticker)
        else:
            self.app.notify("Select a ticker first (use ↑↓ arrows)", severity="warning")

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "delete":
            table = self.query_one("#watchlist-table", MarketTable)
            if table.cursor_row is not None:
                # Get ticker from selected row
                try:
                    ticker = table.get_row_at(table.cursor_row)[0]
                    self.remove_ticker(f"{ticker}.JK")
                except Exception as e:
                    logger.debug(f"Error removing ticker: {e}")
