# 💻 Python API Reference

Programmatic interface for IDX Stock Analyzer.

---

## Table of Contents

- [Quick Start](#quick-start)
- [IDXAnalyzer Class](#idxanalyzer-class)
- [AnalysisResult](#analysisresult)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Quick Start

```python
from idx_analyzer.analyzer import IDXAnalyzer

# Create analyzer
analyzer = IDXAnalyzer("BBCA")

# Fetch data
analyzer.fetch_data(period="1y")

# Get analysis
result = analyzer.analyze()

# Access results
print(f"Price: {result.current_price}")
print(f"Trend: {result.trend}")
print(f"RSI: {result.rsi}")

# Generate chart
chart_path = analyzer.generate_chart(output_path="my_chart.png")
```

---

## IDXAnalyzer Class

### Constructor

```python
IDXAnalyzer(ticker: str, config: Optional[Config] = None)
```

**Parameters:**
- `ticker` (str): Stock ticker (e.g., "BBCA", "TLKM")
- `config` (Config, optional): Custom configuration object

**Examples:**
```python
# Basic usage
analyzer = IDXAnalyzer("BBCA")

# With IDX: prefix
analyzer = IDXAnalyzer("IDX:BBCA")

# With custom config
from idx_analyzer.config import Config
config = Config()
analyzer = IDXAnalyzer("BBCA", config=config)
```

---

### Methods

#### fetch_data()

```python
fetch_data(period: Optional[str] = None) -> bool
```

Fetch historical data and stock info.

**Parameters:**
- `period` (str, optional): Data period. Options: "1mo", "3mo", "6mo", "1y", "2y", "5y"

**Returns:**
- `bool`: True if successful

**Raises:**
- `InvalidTickerError`: If ticker doesn't exist
- `InsufficientDataError`: If not enough data points
- `NetworkError`: If connection fails
- `DataFetchError`: For other fetch errors

**Example:**
```python
analyzer = IDXAnalyzer("BBCA")
analyzer.fetch_data(period="1y")
```

---

#### analyze()

```python
analyze() -> AnalysisResult
```

Perform full technical analysis.

**Returns:**
- `AnalysisResult`: Complete analysis results

**Raises:**
- `AnalysisError`: If analysis fails

**Example:**
```python
result = analyzer.analyze()
print(f"Current Price: {result.current_price}")
print(f"Trend: {result.trend}")
print(f"RSI: {result.rsi}")
```

---

#### generate_chart()

```python
generate_chart(output_path: Optional[str] = None, show: bool = False) -> str
```

Generate technical analysis chart.

**Parameters:**
- `output_path` (str, optional): Custom output filename
- `show` (bool): If True, display chart instead of saving

**Returns:**
- `str`: Absolute path to generated chart

**Raises:**
- `ChartError`: If chart generation fails

**Example:**
```python
# Save to default location
chart_path = analyzer.generate_chart()

# Custom filename
chart_path = analyzer.generate_chart(output_path="my_analysis.png")

# Display instead of save
analyzer.generate_chart(show=True)
```

---

#### generate_rich_report()

```python
generate_rich_report(result: AnalysisResult) -> Group
```

Generate rich text report for CLI display (requires `rich` library).

**Parameters:**
- `result` (AnalysisResult): Analysis result to display

**Returns:**
- `Group`: Rich console group object

**Example:**
```python
from rich.console import Console

result = analyzer.analyze()
report = analyzer.generate_rich_report(result)

console = Console()
console.print(report)
```

---

#### generate_chat_report()

```python
generate_chat_report(result: AnalysisResult) -> str
```

Generate compact summary for messaging apps (Telegram/WhatsApp).

**Parameters:**
- `result` (AnalysisResult): Analysis result

**Returns:**
- `str`: Formatted text summary

**Example:**
```python
result = analyzer.analyze()
chat_summary = analyzer.generate_chat_report(result)
print(chat_summary)
```

---

## AnalysisResult

### Attributes

#### Basic Information

| Attribute | Type | Description |
|-----------|------|-------------|
| `ticker` | `str` | Stock ticker symbol |
| `current_price` | `float` | Current stock price |
| `change_percent` | `float` | Daily change percentage |
| `volume` | `int` | Trading volume |

#### Price Data

| Attribute | Type | Description |
|-----------|------|-------------|
| `week_52_high` | `float` | 52-week high price |
| `week_52_low` | `float` | 52-week low price |

#### Technical Indicators

| Attribute | Type | Description |
|-----------|------|-------------|
| `rsi` | `float` | Relative Strength Index (14-period) |
| `sma_20` | `float` | 20-day Simple Moving Average |
| `sma_50` | `float` | 50-day Simple Moving Average |
| `sma_200` | `Optional[float]` | 200-day Simple Moving Average |
| `bb_middle` | `Optional[float]` | Bollinger Bands middle (SMA 20) |
| `bb_upper` | `Optional[float]` | Bollinger Bands upper |
| `bb_lower` | `Optional[float]` | Bollinger Bands lower |
| `bb_position` | `Optional[str]` | Price position relative to BB |

#### Volume Profile

| Attribute | Type | Description |
|-----------|------|-------------|
| `vp_poc` | `Optional[float]` | Point of Control (highest volume price) |
| `vp_value_area_high` | `Optional[float]` | Value Area upper bound |
| `vp_value_area_low` | `Optional[float]` | Value Area lower bound |
| `vp_total_volume` | `Optional[float]` | Total analyzed volume |

#### Fundamental Data

| Attribute | Type | Description |
|-----------|------|-------------|
| `market_cap` | `Optional[float]` | Market capitalization |
| `pe_ratio` | `Optional[float]` | Price-to-Earnings ratio |
| `dividend_yield` | `Optional[float]` | Dividend yield |

#### Analysis Results

| Attribute | Type | Description |
|-----------|------|-------------|
| `trend` | `str` | Trend classification |
| `recommendation` | `str` | Trading recommendation |
| `summary` | `str` | Brief text summary |
| `support_levels` | `List[SupportResistance]` | Support levels list |
| `resistance_levels` | `List[SupportResistance]` | Resistance levels list |

---

## Configuration

### Loading Configuration

```python
from idx_analyzer.config import load_config, Config

# Load from default locations
config = load_config()

# Load from specific file
config = load_config("/path/to/config.toml")

# Create default config
from idx_analyzer.config import create_default_config
config_path = create_default_config()
```

### Config Classes

#### Config

Main configuration container.

```python
Config(
    chart: ChartConfig = ChartConfig(),
    analysis: AnalysisConfig = AnalysisConfig(),
    network: NetworkConfig = NetworkConfig(),
    display: DisplayConfig = DisplayConfig()
)
```

#### AnalysisConfig

```python
AnalysisConfig(
    default_period: str = "6mo",      # Default analysis period
    rsi_window: int = 14,             # RSI calculation period
    sma_windows: list[int] = [20, 50, 200],  # SMA periods
    bb_window: int = 20,              # Bollinger Bands window
    bb_std: float = 2.0,              # Bollinger Bands std dev
    vp_bins: int = 50                 # Volume Profile bins
)
```

#### ChartConfig

```python
ChartConfig(
    dpi: int = 150,                   # Chart resolution
    width: int = 16,                  # Chart width (inches)
    height: int = 10,                 # Chart height (inches)
    style: str = "default",           # Matplotlib style
    show_grid: bool = True            # Show grid lines
)
```

#### NetworkConfig

```python
NetworkConfig(
    timeout: int = 30,                # Request timeout (seconds)
    max_retries: int = 3,             # Number of retries
    retry_delay: float = 1.0,         # Initial retry delay
    use_cache: bool = True,           # Enable caching
    cache_ttl: int = 86400            # Cache TTL in seconds (24 hours)
)
```

---

## Examples

### Basic Analysis

```python
from idx_analyzer.analyzer import IDXAnalyzer

analyzer = IDXAnalyzer("BBCA")
analyzer.fetch_data(period="6mo")
result = analyzer.analyze()

print(f"{result.ticker}: {result.current_price}")
print(f"Trend: {result.trend}")
print(f"RSI: {result.rsi:.1f}")

if result.support_levels:
    print(f"Support: {result.support_levels[0].level}")

if result.resistance_levels:
    print(f"Resistance: {result.resistance_levels[0].level}")
```

### Batch Analysis

```python
from idx_analyzer.analyzer import IDXAnalyzer

stocks = ["BBCA", "BBRI", "TLKM", "ASII"]

for ticker in stocks:
    try:
        analyzer = IDXAnalyzer(ticker)
        analyzer.fetch_data(period="1y")
        result = analyzer.analyze()
        
        print(f"\n{result.ticker}:")
        print(f"  Price: {result.current_price:,.0f}")
        print(f"  Trend: {result.trend}")
        print(f"  Rec: {result.recommendation}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
```

### Custom Configuration

```python
from idx_analyzer.analyzer import IDXAnalyzer
from idx_analyzer.config import Config, AnalysisConfig

# Create custom config
custom_analysis = AnalysisConfig(
    default_period="1y",
    rsi_window=21,           # Use 21-period RSI
    sma_windows=[10, 30, 100],  # Different SMA periods
    bb_window=21,
    bb_std=2.5
)

config = Config(analysis=custom_analysis)

# Use custom config
analyzer = IDXAnalyzer("BBCA", config=config)
analyzer.fetch_data()
result = analyzer.analyze()
```

### Export Results

```python
import json
from idx_analyzer.analyzer import IDXAnalyzer

analyzer = IDXAnalyzer("BBCA")
analyzer.fetch_data(period="1y")
result = analyzer.analyze()

# Export to JSON
data = {
    "ticker": result.ticker,
    "price": result.current_price,
    "trend": result.trend,
    "rsi": result.rsi,
    "supports": [s.level for s in result.support_levels],
    "resistances": [r.level for r in result.resistance_levels]
}

with open("analysis.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Error Handling

```python
from idx_analyzer.analyzer import IDXAnalyzer
from idx_analyzer.exceptions import (
    InvalidTickerError,
    InsufficientDataError,
    NetworkError
)

analyzer = IDXAnalyzer("INVALID")

try:
    analyzer.fetch_data()
    result = analyzer.analyze()
except InvalidTickerError as e:
    print(f"Invalid ticker: {e.ticker}")
except InsufficientDataError as e:
    print(f"Not enough data: {e.data_points} points available")
except NetworkError as e:
    print(f"Network error: {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## See Also

- [README.md](../README.md) - Project overview
- [USAGE.md](USAGE.md) - Command-line usage
- [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) - Indicator explanations
