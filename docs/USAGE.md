# 📖 User Guide

Complete guide for using IDX Stock Analyzer.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding Output](#understanding-output)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [Supported Stocks](#supported-stocks)

---

## Quick Start

### Prerequisites

- Python 3.13 or higher
- [UV](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/ceroberoz/IDX-Stock-Analyzer.git
cd IDX-Stock-Analyzer

# Install dependencies using UV
uv sync

# Verify installation
uv run idx-analyzer --version
```

> **Don't have UV?** Install it with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Your First Analysis

```bash
# Analyze BBCA (Bank Central Asia)
uv run idx-analyzer BBCA

# Generate a chart
uv run idx-analyzer BBCA --chart

# Analyze with 1 year of data for better SMA 200 accuracy
uv run idx-analyzer BBCA --period 1y --chart
```

---

## Understanding Output

### Terminal Output

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ 📊 Market Intel: TPMA                                    Price: 605 (-0.82%) │
╰──────────────────────────────────────────────────────────────────────────────╯
  🚀 Metric     💎 Value                                             🚦 Status  
  Trend         Bullish                               Bullish (Strong Uptrend)  
  RSI (14)      52.2                                                ⚖️ Neutral  
  Mov. Avgs     📈 >SMA20, 📈 >SMA50, 📈 >SMA200                                
                         🧱 Support & Resistance Zones                          
                                                                                 
  Type                           Level               Distance   Strength        
 ───────────────────────────────────────────────────────────────────────────── 
  🛡️ Support                       600             0.8% below   Weak            
  🛡️ Support                       560             7.4% below   Moderate        
  🧱 Resistance                    620            +2.5% above   Moderate        
  🧱 Resistance                    630            +4.1% above   Moderate        
                                                                                 
🐂 Bullish Vibes Detected! Buyers are in control. 
🎲 Risk/Reward Ratio: 1:3.0
╭─────────────────────────────── ⚡ Action Plan ───────────────────────────────╮
│                       🤔 Bullish. Target: 620 (+2.5%).                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Moving Averages Explained

| MA | Period | What It Tells You |
|----|--------|-------------------|
| **SMA 20** | 20 days | Short-term trend direction |
| **SMA 50** | 50 days | Medium-term trend strength |
| **SMA 200** | 200 days | Long-term trend / major support-resistance |

**Golden Cross**: SMA 50 crosses above SMA 200 → **Bullish signal** 📈  
**Death Cross**: SMA 50 crosses below SMA 200 → **Bearish signal** 📉

### Recommendation Types

- **BUY** - Strong uptrend, consider dips to support
- **SELL/AVOID** - Downtrend active, wait for reversal
- **REDUCE** - Very overbought, consider taking profits
- **WATCH** - Very oversold, reversal may be near
- **HOLD/BUY** - Bullish trend, use support for entry
- **WAIT** - No clear directional bias

---

## Command Reference

```bash
uv run idx-analyzer <TICKER> [OPTIONS]
```

### Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--period` | `-p` | Historical data period | `--period 1y` |
| `--export` | `-e` | Export format (csv/json) | `--export json` |
| `--output` | `-o` | Custom output filename | `--output my_analysis.json` |
| `--chart` | `-c` | Generate technical chart | `--chart` |
| `--chart-output` | | Custom chart filename | `--chart-output bbc.png` |
| `--chat` | | Generate compact chat report | `--chat` |
| `--config` | | Custom configuration file | `--config myconfig.toml` |
| `--init-config` | | Create default config file | `--init-config` |
| `--sentiment` | | Analyze news sentiment (FinBERT) | `--sentiment` |
| `--sentiment-vader` | | Lightweight VADER sentiment | `--sentiment-vader` |
| `--cache-info` | | Show HTTP cache information | `--cache-info` |
| `--clear-cache` | | Clear HTTP cache | `--clear-cache` |
| `--quiet` | `-q` | Minimal output for scripting | `--quiet` |
| `--version` | `-v` | Show version | `--version` |

### Period Options

- `1mo` - 1 month
- `3mo` - 3 months
- `6mo` - 6 months (default)
- `1y` - 1 year (recommended for SMA 200)
- `2y` - 2 years
- `5y` - 5 years

### Usage Examples

```bash
# Basic analysis
uv run idx-analyzer BBCA

# Analyze with 1 year data and generate chart
uv run idx-analyzer BBRI --period 1y --chart

# Export to JSON
uv run idx-analyzer TLKM --export json --output tlkm_analysis.json

# Batch analysis (scripting mode)
uv run idx-analyzer ASII --quiet

# Custom chart filename
uv run idx-analyzer UNVR -c --chart-output unvr_chart.png

# Create default configuration file
uv run idx-analyzer BBCA --init-config

# Use custom configuration file
uv run idx-analyzer BBCA --config myconfig.toml
```

### Output File Organization

Charts and exports are automatically organized by ticker and date:

```
charts/
└── BBCA/
    └── 2026-02-14/
        └── BBCA_chart.png

exports/
└── BBCA/
    └── 2026-02-14/
        └── BBCA_analysis.json
```

This keeps your workspace clean when analyzing multiple stocks over time.

---

## Configuration

IDX Analyzer supports TOML configuration files for customizing default behavior.

### Configuration Locations

The tool looks for config files in this order:
1. Path specified with `--config`
2. `~/.config/idx-analyzer/config.toml`
3. `~/.idx-analyzer.toml`
4. `./idx-analyzer.toml` (current directory)

### Example Configuration

```toml
[analysis]
default_period = "6mo"        # Default: 6mo, Options: 1mo, 3mo, 6mo, 1y, 2y, 5y
rsi_window = 14               # RSI calculation period
sma_windows = [20, 50, 200]   # SMA periods
bb_window = 20                # Bollinger Bands window
bb_std = 2.0                  # Bollinger Bands standard deviations
vp_bins = 50                  # Volume Profile bins

[network]
timeout = 30                  # Request timeout (seconds)
max_retries = 3               # Number of retries for failed requests
retry_delay = 1.0             # Initial retry delay (seconds)
use_cache = true              # Enable request caching
cache_ttl = 86400             # Cache TTL in seconds (24 hours)

[chart]
dpi = 150                     # Chart resolution
width = 16                    # Chart width (inches)
height = 10                   # Chart height (inches)
style = "default"             # Matplotlib style
show_grid = true              # Show grid lines

[display]
color_output = true           # Enable colored terminal output
verbose = false               # Verbose output mode
```

### Creating a Config File

```bash
# Create default config at ~/.idx-analyzer.toml
uv run idx-analyzer BBCA --init-config

# Or copy the example and customize
cp idx-analyzer.toml.example ~/.idx-analyzer.toml
```

---

## Supported Stocks

Any stock listed on the Indonesia Stock Exchange (IDX). Use the ticker symbol without the `.JK` suffix.

### Popular Tickers by Sector

| Sector | Tickers |
|--------|---------|
| **Banking** | BBCA, BBRI, BMRI, BBNI, BRIS |
| **Telco** | TLKM, ISAT, EXCL, FREN |
| **Consumer** | UNVR, ICBP, MYOR, GGRM, KAEF |
| **Mining** | ADRO, ITMG, PTBA, ANTM, INCO |
| **Property** | SMRA, PWON, CTRA, BSDE, APLN |
| **Infrastructure** | ASII, JSMR, CMNP, JPFA |
| **Technology** | GOTO, BELI, BUKA, MCAS |

> **Tip:** You can also use `IDX:` prefix (e.g., `IDX:BBCA`)

---

## Chat Reports

Generate instant, copy-paste ready summaries for messaging apps with `--chat`:

```
📊 *BBCA Daily Update*
🔴 Price: 7,200 (-1.71%)
🌊 Trend: 🐻 Bearish

📉 *Tech Stats:*
• RSI: 38.8
• Vol: 356.0M

🎯 *Key Levels:*
• 🧱 Res: 7,500
• 🛡️ Sup: 7,200

💡 *Outlook:*
Bearish. Support at 7,200 (0.0% below).

🚨 *Action:* WAIT
```

---

## Troubleshooting

### Common Issues

**"Invalid or unknown ticker"**
- Check the ticker spelling (e.g., 'BBCA' not 'BB CA')
- Use the stock code without '.JK' suffix
- Verify the stock is listed on IDX
- Try: BBCA, BBRI, TLKM, ASII, UNVR

**"Could not fetch data" / Network errors**
- Check your internet connection
- Yahoo Finance may be temporarily unavailable
- The tool will automatically retry up to 3 times
- Try again in a few moments

**"Insufficient data"**
- Try a longer period: `--period 1y` or `--period 2y`
- The stock may be newly listed
- Try a different ticker

**Charts not generating**
- Ensure you have write permissions in the directory
- Try specifying a full path: `--chart-output /path/to/chart.png`
- Check that you have sufficient disk space

---

*For more information, see [README.md](../README.md)*
