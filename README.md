# 📈 IDX Stock Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/IDX-Indonesia%20Stock%20Exchange-red.svg" alt="IDX">
</p>

A powerful command-line tool for technical analysis of Indonesian stocks (IDX). Get instant insights on support/resistance, trends, and actionable recommendations.

---

## ✨ Features

- 🔍 **Smart Support/Resistance** - Auto-detect key price levels
- 📊 **Multi-Timeframe Analysis** - SMA 20/50/200, RSI, MACD, Bollinger Bands
- 📈 **Intraday Charts** - 1m, 5m, 15m, 30m, 1h intervals (with Yahoo limits)
- 🎛️ **TUI Interface** - Bloomberg-style terminal with keyboard shortcuts
- 📁 **Export Options** - CSV, JSON, Excel with formatted sheets
- 🕯️ **Pattern Detection** - Doji, Hammer, Engulfing, Morning Star
- 📰 **News Sentiment** - FinBERT with Indonesian hybrid (v1.4)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [UV](https://github.com/astral-sh/uv) package manager

### Installation
```bash
git clone https://github.com/ceroberoz/IDX-Stock-Analyzer.git
cd IDX-Stock-Analyzer
uv sync
```

### Usage

```bash
# Basic analysis
uv run idx-analyzer BBCA

# Intraday analysis (1m, 5m, 15m, 30m, 1h)
uv run idx-analyzer BBCA --interval 5m --period 5d

# Generate chart
uv run idx-analyzer BBCA --chart --all

# Export to Excel
uv run idx-analyzer BBCA --export excel

# Launch TUI mode
uv run idx-analyzer --tui
```

---

## 📊 Example Output

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ 📊 Market Intel: BBCA                                    Price: 9,250 (+1.2%) │
╰──────────────────────────────────────────────────────────────────────────────╯
  🚀 Metric     💎 Value                                             🚦 Status  
  Trend         Bullish                               Bullish (Strong Uptrend)  
  RSI (14)      52.2                                                ⚖️ Neutral  
  Mov. Avgs     📈 >SMA20, 📈 >SMA50, 📈 >SMA200                                
                         🧱 Support & Resistance Zones                          
  Type                           Level               Distance   Strength        
 ───────────────────────────────────────────────────────────────────────────── 
  🛡️ Support                    9,100              1.6% below   Moderate        
  🧱 Resistance                  9,500              2.7% above   Strong          

🐂 Bullish Vibes Detected! Buyers are in control.
╭─────────────────────────────── ⚡ Action Plan ───────────────────────────────╮
│                       🤔 Bullish. Target: 9,500 (+2.7%).                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [📚 User Guide](docs/USAGE.md) | Complete command reference |
| [📊 Implementation Status](docs/IMPLEMENTATION_STATUS.md) | Feature status & roadmap |
| [🗺️ Strategic Roadmap](docs/STRATEGIC_ROADMAP.md) | Product roadmap & epics |
| [💻 API Reference](docs/API.md) | Developer documentation |
| [🤝 Contributing](CONTRIBUTING.md) | Contribution guidelines |

---

## 🛠️ Supported Stocks

Any stock on the Indonesia Stock Exchange (IDX). Popular tickers:

| Sector | Tickers |
|--------|---------|
| **Banking** | BBCA, BBRI, BMRI, BBNI, BRIS |
| **Telco** | TLKM, ISAT, EXCL, FREN |
| **Consumer** | UNVR, ICBP, MYOR, GGRM, KAEF |
| **Mining** | ADRO, ITMG, PTBA, ANTM, INCO |
| **Technology** | GOTO, BELI, BUKA, MCAS |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git checkout development
git checkout -b feature/your-feature
# Make changes
uv run ruff format .
uv run ruff check .
git commit -m "feat: add your feature"
```

---

## 🙏 Credits

- **[Yahoo Finance](https://finance.yahoo.com/)** - Stock data provider
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Python Yahoo Finance library

---

## 📄 License

MIT License. See [LICENSE](LICENSE) file.

---

<p align="center">
  <b>Happy Trading! 📈🚀</b><br>
  <sub>Built with ❤️ for the Indonesian trading community</sub>
</p>
