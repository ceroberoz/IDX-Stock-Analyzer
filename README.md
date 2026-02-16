# 📈 IDX Stock Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/IDX-Indonesia%20Stock%20Exchange-red.svg" alt="IDX">
</p>

A powerful, user-friendly command-line tool for technical analysis of Indonesian stocks listed on the **Indonesia Stock Exchange (IDX)**. Get instant insights on support/resistance levels, trend analysis, moving averages, and actionable trading recommendations with beautiful visualizations.

---

## ✨ Features

- 🔍 **Smart Support & Resistance Detection** - Automatically identifies key price levels
- 📊 **Multi-Timeframe Trend Analysis** - SMA 20, 50, 200 with Golden/Death Cross detection
- 📈 **Enhanced Technical Charts** - Bollinger Bands, Volume Profile, RSI, SMA lines
- 📰 **News Sentiment Analysis** - FinBERT with Indonesian hybrid (v1.4) optimized for 2025-2026 market trends (FCA, Danantara, MSCI/Moody's shifts)
- 📁 **Export Options** - Save analysis to CSV or JSON
- 🎨 **Modern UI** - Beautiful dashboard with emoji-rich insights
- 📱 **Chat-Ready Reports** - Instant summaries for Telegram/WhatsApp

---

## 📂 Project Structure

```
IDX-Stock-Analyzer/
├── idx_analyzer/           # Main package
│   ├── __init__.py
│   ├── analyzer.py         # Core analysis engine
│   ├── cache.py            # HTTP cache management
│   ├── chart.py            # Unified chart generation (standard & executive)
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── exceptions.py       # Custom exceptions
│   └── sentiment.py        # News sentiment analysis
├── charts/                 # Generated charts (gitignored)
├── exports/                # Generated exports (gitignored)
├── docs/                   # Documentation
├── examples/               # Example scripts
├── tests/                  # Test files
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── .python-version         # Python version pin
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/ceroberoz/IDX-Stock-Analyzer.git
cd IDX-Stock-Analyzer
uv sync
```

### Your First Analysis

```bash
# Analyze BBCA (Bank Central Asia)
uv run idx-analyzer BBCA

# Generate a chart
uv run idx-analyzer BBCA --chart

# Executive dashboard style (high-end layout)
uv run idx-analyzer BBCA --chart --chart-style executive

# 1 year of data for SMA 200
uv run idx-analyzer BBCA --period 1y --chart
```

### Output Files

Generated files are organized by ticker and date:
- **Charts**: `charts/BBCA/2026-02-14/BBCA_chart.png`
- **Exports**: `exports/BBCA/2026-02-14/BBCA_analysis.json`

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [📚 User Guide](docs/USAGE.md) | Complete command reference and configuration |
| [📊 Technical Analysis](docs/TECHNICAL_ANALYSIS.md) | Understanding indicators and signals |
| [💻 Python API](docs/API.md) | Programmatic interface documentation |
| [🗺️ Roadmap](docs/ROADMAP.md) | Development phases and Yahoo Finance API capabilities |
| [🤝 Contributing](CONTRIBUTING.md) | How to contribute and development workflow |

---

## 📈 Example Output

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

🐂 Bullish Vibes Detected! Buyers are in control. 
🎲 Risk/Reward Ratio: 1:3.0
╭─────────────────────────────── ⚡ Action Plan ───────────────────────────────╮
│                       🤔 Bullish. Target: 620 (+2.5%).                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 🛠️ Supported Stocks

Any stock listed on the Indonesia Stock Exchange (IDX). Popular tickers:

| Sector | Tickers |
|--------|---------|
| **Banking** | BBCA, BBRI, BMRI, BBNI, BRIS |
| **Telco** | TLKM, ISAT, EXCL, FREN |
| **Consumer** | UNVR, ICBP, MYOR, GGRM, KAEF |
| **Mining** | ADRO, ITMG, PTBA, ANTM, INCO |
| **Technology** | GOTO, BELI, BUKA, MCAS |

---

## 🗺️ Development Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the complete development plan.

**Current Version:** 1.0.0 | **Yahoo Finance API Utilization:** ~30%

### Phase 1: Core Enhancements (Q1 2026)
- Intraday analysis (5m, 15m, 30m, 1h intervals)
- Batch portfolio analysis
- Excel export
- Dividend tracking

### Phase 2: Advanced Analytics (Q2-Q3 2026)
- Backtesting engine
- Real-time price alerts
- Fundamental screener
- Options flow analysis

### Phase 3: Institutional-Grade (Q4 2026)
- Portfolio optimization
- Volatility surface
- ✅ **News sentiment** - FinBERT with Indonesian hybrid enhancement
- Analyst recommendations

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start:
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

### Data Providers
- **[Yahoo Finance](https://finance.yahoo.com/)** - Real-time and historical stock data
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Python library for Yahoo Finance

### Tools & Libraries
- **UV** - Fast Python package manager
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Rich** - Beautiful terminal output

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/ceroberoz/IDX-Stock-Analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ceroberoz/IDX-Stock-Analyzer/discussions)

---

<p align="center">
  <b>Happy Trading! 📈🚀</b>
</p>

<p align="center">
  <sub>Built with ❤️ for the Indonesian trading community</sub>
</p>
