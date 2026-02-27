"""
IDX Terminal - Main TUI Application
Bloomberg-style terminal for Indonesian Stock Exchange
"""

import logging


from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Static

from .screens.chart_screen import ChartScreen
from .screens.comparison_screen import ComparisonScreen
from .screens.market_screen import MarketScreen
from .screens.stock_screen import StockScreen
from .screens.watchlist_screen import WatchlistScreen
from .themes import get_catppuccin_theme
from .widgets.command_palette import CommandPalette

logger = logging.getLogger(__name__)

class IDXTerminalApp(App):
    """IDX Terminal - Bloomberg-style TUI for Indonesian stocks"""

    CSS = """
    /* Main Layout */
    Screen {
        layout: vertical;
    }

    /* Header Styling */
    Header {
        dock: top;
        height: 3;
        background: $surface;
        color: $primary;
    }

    /* Command Input Area */
    #command-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }

    #command-input {
        width: 100%;
        height: 1;
        border: none;
        padding: 0 2;
        color: $text;
        background: transparent;
    }

    #command-input:focus {
        border: none;
    }

    #command-prompt {
        width: auto;
        color: $primary;
        text-style: bold;
    }

    /* Footer */
    Footer {
        dock: bottom;
        height: 1;
        background: $surface;
    }

    /* Content Areas */
    #main-content {
        width: 100%;
        height: 1fr;
        overflow: auto;
    }

    /* Panel Layout */
    .panel {
        border: solid $surface;
        padding: 1;
    }

    .panel-title {
        color: $primary;
        text-style: bold;
        height: 1;
    }

    /* Stock Ticker Colors */
    .price-up {
        color: $success;
    }

    .price-down {
        color: $error;
    }

    .price-unchanged {
        color: $text;
    }

    /* Data Tables */
    DataTable {
        background: $surface;
        border: none;
    }

    DataTable > .datatable--header {
        background: $surface;
        color: $primary;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: $primary 30%;
    }

    /* Scrollbars */
    Scrollbar {
        background: $surface;
        width: 1;
    }

    /* Loading Spinner */
    LoadingIndicator {
        color: $primary;
    }
    """

    BINDINGS = [
        # Navigation
        Binding("q,ctrl+c", "quit", "Quit", show=True, key_display="Q"),
        Binding("ctrl+l", "focus_command", "Command", show=True, key_display="^L"),
        Binding("escape", "clear_command", "Clear", show=False),
        # Screens
        Binding("f1", "screen_market", "Market", show=True, key_display="F1"),
        Binding("f2", "screen_watchlist", "Watchlist", show=True, key_display="F2"),
        Binding("f3", "screen_stock", "Stock", show=True, key_display="F3"),
        Binding("f4", "screen_compare", "Compare", show=True, key_display="F4"),
        # Actions
        Binding("r", "refresh", "Refresh", show=True),
        Binding("slash", "search", "Search", show=True),
        Binding("?", "help", "Help", show=False),
    ]

    # Register screens so switch_screen works by name
    SCREENS = {
        "market-screen": MarketScreen,
        "watchlist-screen": WatchlistScreen,
    }

    # Reactive state
    current_screen_name: reactive[str] = reactive("market")
    command_history: reactive[list] = reactive(list)
    command_index: reactive[int] = reactive(0)

    # Theme registration
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_theme(get_catppuccin_theme())
        self.theme = "catppuccin-mocha"

    def compose(self) -> ComposeResult:
        """Compose the main app layout"""
        yield Header(show_clock=True)

        with Vertical(id="main-content"):
            # Content will be managed by screens
            pass

        # Bloomberg-style command bar at bottom
        with Horizontal(id="command-bar"):
            yield Static("> ", id="command-prompt")
            yield Input(
                placeholder="Type command (e.g., BBCA, BBCA GP, HELP)",
                id="command-input",
            )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize app on mount"""
        self.title = "IDX Terminal"
        self.sub_title = "v1.1.0 | Bloomberg-style Market Monitor"

        # Push initial screen (push_screen properly sets up the callback stack)
        self.push_screen("market-screen")
        self.current_screen_name = "market"

        # Focus command input
        self.query_one("#command-input", Input).focus()

    # ============================================================================
    # Command Handling
    # ============================================================================

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input submission"""
        command = event.value.strip().upper()
        if not command:
            return

        # Add to history
        self.command_history.append(command)
        self.command_index = len(self.command_history)

        # Process command
        self.process_command(command)

        # Clear input
        event.input.value = ""

    def process_command(self, command: str) -> None:
        """Process Bloomberg-style commands"""
        parts = command.split()
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Navigation commands
        if cmd == "HELP":
            self.action_help()
        elif cmd == "QUIT" or cmd == "EXIT":
            self.action_quit()
        elif cmd == "REFRESH" or cmd == "R":
            self.action_refresh()

        # Screen navigation
        elif cmd == "MARKET" or cmd == "F1":
            self.action_screen_market()
        elif cmd == "WATCHLIST" or cmd == "F2":
            self.action_screen_watchlist()

        # Stock commands
        elif cmd == "STOCK" and args:
            self.show_stock(args[0])
        elif cmd == "COMPARE" or cmd == "COMP":
            tickers = (
                [f"{t}.JK" if not t.endswith(".JK") else t for t in args]
                if args
                else None
            )
            self.show_comparison(tickers)
        elif cmd.endswith(".JK") or len(cmd) == 4 and cmd.isalpha():
            # Direct ticker input: BBCA or BBCA.JK
            ticker = cmd if cmd.endswith(".JK") else f"{cmd}.JK"
            if args and args[0] == "GP":
                # BBCA GP - Show chart
                self.show_stock_chart(ticker)
            else:
                self.show_stock(ticker)

        else:
            self.notify(f"Unknown command: {command}", severity="warning")

    # ============================================================================
    # Actions
    # ============================================================================

    def action_focus_command(self) -> None:
        """Focus the command input"""
        self.query_one("#command-input", Input).focus()

    def action_clear_command(self) -> None:
        """Clear command input and unfocus"""
        input_widget = self.query_one("#command-input", Input)
        input_widget.value = ""
        self.screen.focus_next()

    def action_screen_market(self) -> None:
        """Switch to market overview screen"""
        if self.current_screen_name == "market":
            return
        self.switch_screen("market-screen")
        self.current_screen_name = "market"
        self.title = "IDX Terminal - Market Overview"

    def action_screen_watchlist(self) -> None:
        """Switch to watchlist screen"""
        if self.current_screen_name == "watchlist":
            return
        self.switch_screen("watchlist-screen")
        self.current_screen_name = "watchlist"
        self.title = "IDX Terminal - Watchlist"

    def action_screen_stock(self) -> None:
        """Switch to stock screen (prompt for ticker)"""
        self.query_one("#command-input", Input).focus()
        self.notify("Enter ticker symbol (e.g., BBCA)", severity="information")

    def action_screen_compare(self) -> None:
        """Switch to comparison screen"""
        self.show_comparison()

    def action_refresh(self) -> None:
        """Refresh current screen data"""
        current = self.screen
        if hasattr(current, "refresh_data"):
            current.refresh_data()
            self.notify("Data refreshed", severity="information")

    def action_search(self) -> None:
        """Open search/command palette"""
        self.push_screen(CommandPalette())

    def action_help(self) -> None:
        """Show help dialog"""
        help_text = """
        [b]IDX Terminal Commands[/b]

        [b]Navigation:[/b]
        • F1 / MARKET    - Market overview screen
        • F2 / WATCHLIST - Watchlist screen
        • F3 / STOCK     - Stock detail screen
        • F4 / COMPARE   - Multi-ticker comparison
        • Ctrl+L         - Focus command bar
        • Q / Ctrl+C     - Quit

        [b]Stock Commands:[/b]
        • TICKER         - Show stock profile (e.g., BBCA)
        • TICKER GP      - Show stock chart
        • COMPARE T1 T2  - Compare tickers

        [b]Examples:[/b]
        • BBCA           - View BBCA profile
        • BBRI GP        - View BBRI chart
        • COMP BBCA BBRI - Compare vs BBRI

        [b]General:[/b]
        • R / REFRESH    - Refresh data
        • /              - Search
        • ? / HELP       - Show this help
        """
        self.notify(help_text, title="Help", timeout=10)

    # ============================================================================
    # Stock Display Methods
    # ============================================================================

    def show_stock(self, ticker: str) -> None:
        """Show stock profile screen"""
        # Clean ticker
        ticker = ticker.upper().replace("IDX:", "").replace(".JK", "")
        ticker_jk = f"{ticker}.JK"

        screen = StockScreen(ticker=ticker_jk, id=f"stock-{ticker}")
        self.push_screen(screen)
        self.current_screen_name = f"stock-{ticker}"
        self.title = f"IDX Terminal - {ticker}"

    def show_stock_chart(self, ticker: str) -> None:
        """Show stock chart screen"""
        ticker = ticker.upper().replace(".JK", "")
        ticker_jk = f"{ticker}.JK"

        # Remove existing screen if present
        screen_id = f"chart-{ticker}"
        try:
            self.uninstall_screen(screen_id)
        except KeyError:
            logger.debug(f"Screen {screen_id} not found for uninstall")
        except KeyError:
            logger.debug(f"Screen {screen_id} not found for uninstall")

        from .screens.chart_screen import ChartScreen

        screen = ChartScreen(ticker=ticker_jk, id=screen_id)
        self.push_screen(screen)
        self.current_screen_name = screen_id
        self.title = f"IDX Terminal - {ticker} Chart"

    def show_comparison(self, tickers: list = None) -> None:
        """Show comparison screen"""
        screen_id = "comparison-screen"
        try:
            self.uninstall_screen(screen_id)
        except KeyError:
            pass

        from .screens.comparison_screen import ComparisonScreen

        screen = ComparisonScreen(tickers=tickers, id=screen_id)
        self.push_screen(screen)
        self.current_screen_name = "comparison"
        self.title = "IDX Terminal - Comparison"


# Entry point
def run_tui():
    """Run the TUI application"""
    app = IDXTerminalApp()
    app.run()


if __name__ == "__main__":
    run_tui()
