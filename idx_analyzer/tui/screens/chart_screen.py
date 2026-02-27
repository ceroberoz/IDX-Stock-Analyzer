"""
Interactive Chart Screen for IDX Terminal
Uses textual-plotext for in-terminal charts
"""

import yfinance as yf
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Label, Select, Static
from textual_plotext import PlotextPlot

from ..themes import COLORS


class ChartScreen(Screen):
    """Interactive chart screen with multiple timeframes and indicators"""

    CSS = """
    #chart-container {
        layout: vertical;
        height: 100%;
        padding: 1;
    }

    #chart-header {
        height: auto;
        layout: horizontal;
    }

    #chart-title {
        width: 1fr;
        color: $primary;
        text-style: bold;
    }

    #chart-controls {
        height: auto;
        layout: horizontal;
        margin: 1 0;
    }

    #interval-select, #period-select {
        width: 15;
        margin-right: 1;
    }

    #refresh-btn {
        width: auto;
    }

    #chart-plot {
        height: 1fr;
        border: solid $surface;
    }

    #indicators-panel {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }

    .indicator-toggle {
        width: auto;
        margin-right: 1;
    }

    #footer-hint {
        height: 1;
        color: $text;
        text-style: dim;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("1", "interval_1m", "1m"),
        ("5", "interval_5m", "5m"),
        ("!", "interval_15m", "15m"),
        ("@", "interval_30m", "30m"),
        ("#", "interval_1h", "1h"),
        ("d", "interval_1d", "1d"),
        ("escape", "back", "Back"),
    ]

    # Reactive state
    ticker: str = ""
    interval: reactive[str] = reactive("1d")
    period: reactive[str] = reactive("6mo")
    show_sma: reactive[bool] = reactive(True)
    show_volume: reactive[bool] = reactive(True)
    chart_data: reactive[dict] = reactive(dict)

    def __init__(self, ticker: str = "BBCA.JK", **kwargs):
        super().__init__(**kwargs)
        self.ticker = ticker.upper()
        self._loading = False

    def compose(self) -> ComposeResult:
        """Compose chart screen"""
        with Vertical(id="chart-container"):
            # Header
            with Horizontal(id="chart-header"):
                yield Label(
                    f"📊 {self.ticker.replace('.JK', '')} Chart", id="chart-title"
                )

            # Controls
            with Horizontal(id="chart-controls"):
                yield Select(
                    options=[
                        ("1m", "1m"),
                        ("5m", "5m"),
                        ("15m", "15m"),
                        ("30m", "30m"),
                        ("1h", "1h"),
                        ("1d", "1d"),
                    ],
                    value="1d",
                    id="interval-select",
                    prompt="Interval",
                )
                yield Select(
                    options=[
                        ("1d", "1d"),
                        ("5d", "5d"),
                        ("1mo", "1mo"),
                        ("3mo", "3mo"),
                        ("6mo", "6mo"),
                        ("1y", "1y"),
                    ],
                    value="6mo",
                    id="period-select",
                    prompt="Period",
                )
                yield Button("🔄 Refresh", id="refresh-btn", variant="primary")

            # Chart
            yield PlotextPlot(id="chart-plot")

            # Indicators
            with Horizontal(id="indicators-panel"):
                yield Button("SMA", id="btn-sma", variant="primary")
                yield Button("Volume", id="btn-volume", variant="primary")
                yield Button("RSI", id="btn-rsi", variant="default")
                yield Button("MACD", id="btn-macd", variant="default")
                yield Button("🔙 Back", id="btn-back", variant="default")

            yield Static(
                "[dim]R=Refresh | 1=1m | 5=5m | !=15m | @=30m | #=1h | D=Daily | Esc=Back[/dim]",
                id="footer-hint",
            )

    def on_mount(self) -> None:
        """Load initial chart data"""
        self.refresh_chart()

    async def refresh_chart(self) -> None:
        """Fetch and render chart data"""
        if self._loading:
            return

        self._loading = True

        try:
            # Fetch data
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period=self.period, interval=self.interval)

            if hist.empty:
                self.notify("No data available", severity="warning")
                return

            self.chart_data = {
                "dates": list(range(len(hist))),
                "closes": hist["Close"].tolist(),
                "opens": hist["Open"].tolist(),
                "highs": hist["High"].tolist(),
                "lows": hist["Low"].tolist(),
                "volumes": hist["Volume"].tolist(),
            }

            # Calculate SMAs
            if len(hist) >= 20:
                self.chart_data["sma20"] = hist["Close"].rolling(20).mean().tolist()
            if len(hist) >= 50:
                self.chart_data["sma50"] = hist["Close"].rolling(50).mean().tolist()

            self.render_plotext_chart()

        except Exception as e:
            self.notify(f"Chart error: {e}", severity="error")
        finally:
            self._loading = False

    def render_plotext_chart(self) -> None:
        """Render chart using plotext"""
        if not self.chart_data:
            return

        plot = self.query_one("#chart-plot", PlotextPlot)

        # Clear previous plot
        plot.clear()

        # Get data
        dates = self.chart_data["dates"]
        closes = self.chart_data["closes"]

        # Create figure
        plot.plt.title(f"{self.ticker.replace('.JK', '')} - {self.interval}")
        plot.plt.xlabel("Time")
        plot.plt.ylabel("Price (IDR)")

        # Plot price
        plot.plt.plot(dates, closes, label="Close", color="cyan")

        # Add SMA if enabled
        if self.show_sma:
            if "sma20" in self.chart_data:
                plot.plt.plot(
                    dates, self.chart_data["sma20"], label="SMA20", color="yellow"
                )
            if "sma50" in self.chart_data:
                plot.plt.plot(
                    dates, self.chart_data["sma50"], label="SMA50", color="magenta"
                )

        # Add volume subplot if enabled
        if self.show_volume and "volumes" in self.chart_data:
            plot.plt.subplot(2, 1)
            plot.plt.bar(
                dates, self.chart_data["volumes"], label="Volume", color="blue"
            )

        plot.refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle interval/period changes"""
        if event.select.id == "interval-select":
            self.interval = event.value
            # Adjust period for intraday
            if self.interval in ["1m", "5m", "15m", "30m", "1h"]:
                if self.period in ["6mo", "1y", "2y", "5y"]:
                    self.period = "1mo"
                    period_select = self.query_one("#period-select", Select)
                    period_select.value = "1mo"
        elif event.select.id == "period-select":
            self.period = event.value

        self.refresh_chart()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        btn_id = event.button.id

        if btn_id == "refresh-btn":
            self.action_refresh()
        elif btn_id == "btn-sma":
            self.show_sma = not self.show_sma
            event.button.variant = "primary" if self.show_sma else "default"
            self.render_plotext_chart()
        elif btn_id == "btn-volume":
            self.show_volume = not self.show_volume
            event.button.variant = "primary" if self.show_volume else "default"
            self.render_plotext_chart()
        elif btn_id == "btn-rsi":
            self.notify("RSI overlay - Coming soon", severity="information")
        elif btn_id == "btn-macd":
            self.notify("MACD overlay - Coming soon", severity="information")
        elif btn_id == "btn-back":
            self.action_back()

    def action_refresh(self) -> None:
        """Refresh chart"""
        self.run_worker(self.refresh_chart())

    def action_interval_1m(self) -> None:
        """Switch to 1m interval"""
        self.interval = "1m"
        self.period = "1d"
        self.query_one("#interval-select", Select).value = "1m"
        self.query_one("#period-select", Select).value = "1d"
        self.refresh_chart()

    def action_interval_5m(self) -> None:
        """Switch to 5m interval"""
        self.interval = "5m"
        self.period = "5d"
        self.query_one("#interval-select", Select).value = "5m"
        self.query_one("#period-select", Select).value = "5d"
        self.refresh_chart()

    def action_interval_15m(self) -> None:
        """Switch to 15m interval"""
        self.interval = "15m"
        self.period = "1mo"
        self.query_one("#interval-select", Select).value = "15m"
        self.query_one("#period-select", Select).value = "1mo"
        self.refresh_chart()

    def action_interval_30m(self) -> None:
        """Switch to 30m interval"""
        self.interval = "30m"
        self.period = "1mo"
        self.query_one("#interval-select", Select).value = "30m"
        self.query_one("#period-select", Select).value = "1mo"
        self.refresh_chart()

    def action_interval_1h(self) -> None:
        """Switch to 1h interval"""
        self.interval = "1h"
        self.period = "3mo"
        self.query_one("#interval-select", Select).value = "1h"
        self.query_one("#period-select", Select).value = "3mo"
        self.refresh_chart()
    def action_interval_1d(self) -> None:
        """Switch to 1d interval"""
        self.interval = "1d"
        self.period = "6mo"
        self.query_one("#interval-select", Select).value = "1d"
        self.query_one("#period-select", Select).value = "6mo"
        self.refresh_chart()

    def action_back(self) -> None:
        """Go back"""
        self.app.pop_screen()
