# 🗺️ Development Roadmap

This roadmap outlines the development phases for IDX Stock Analyzer, leveraging the full capabilities of the Yahoo Finance API via `yfinance`.

---

## Current Status

**Current Version:** 0.1.0  
**Last Updated:** 2026-02-14  
**Yahoo Finance API Utilization:** ~30%

---

## Development Phases

### Phase 1: Core Enhancements (Immediate)

**Goal:** Expand current capabilities with low-effort, high-value features

| Feature | Yahoo Finance API Endpoint | Status | Priority |
|---------|---------------------------|--------|----------|
| **Intraday Analysis** | `history(interval="15m")` | Planned | High |
| **Batch Portfolio Analysis** | `yf.download()` | Planned | High |
| **Dividend Calendar** | `Ticker.dividends` | Planned | Medium |
| **Stock Splits** | `Ticker.splits` | Planned | Medium |
| **Excel Export** | Current data + `pd.ExcelWriter` | Planned | Medium |

**Deliverables:**
- [ ] Add `--interval` flag (1m, 5m, 15m, 30m, 1h, 1d)
- [ ] Multi-ticker analysis: `idx-analyzer BBCA BBRI TLKM`
- [ ] Dividend yield tracking and alerts
- [ ] Export to Excel format
- [ ] Intraday chart generation

---

### Phase 2: Advanced Analytics (Medium-Term)

**Goal:** Implement sophisticated analysis features using Yahoo Finance data

| Feature | Yahoo Finance API Endpoint | Status | Priority |
|---------|---------------------------|--------|----------|
| **Backtesting Engine** | `history()` historical data | Planned | High |
| **Real-time Price Alerts** | WebSocket streaming | Planned | High |
| **Fundamental Screener** | `screener.EquityQuery` | Planned | High |
| **Options Flow Analysis** | `option_chain()` | Planned | Medium |
| **Earnings Tracker** | `earnings_dates` | Planned | Medium |
| **Sector Rotation** | `Sector` performance | Planned | Low |

**Deliverables:**
- [ ] Backtesting module with strategy definition
- [ ] Alert system (email/webhook) for price thresholds
- [ ] Stock screener (find undervalued stocks)
- [ ] Options chain viewer with Greeks
- [ ] Earnings calendar integration
- [ ] Sector comparison tool

---

### Phase 3: Institutional-Grade Features (Advanced)

**Goal:** Professional-level tools for serious traders

| Feature | Yahoo Finance API Endpoint | Status | Priority |
|---------|---------------------------|--------|----------|
| **Portfolio Optimization** | Multi-ticker correlation | Planned | High |
| **Volatility Surface** | Options across expirations | Planned | Medium |
| **Insider Transaction Tracking** | Holders data | Planned | Medium |
| **Institutional Ownership** | `institutional_holders` | Planned | Low |
| **News Sentiment** | `Ticker.news` | Planned | Low |
| **Analyst Recommendations** | `Ticker.recommendations` | Planned | Low |

**Deliverables:**
- [ ] Portfolio correlation and Sharpe ratio analysis
- [ ] Volatility analysis (historical vs implied)
- [ ] Smart money flow indicators
- [ ] News sentiment scoring
- [ ] Analyst rating aggregation

---

## Yahoo Finance API Capabilities Overview

### Data Types Currently Used
- ✅ Historical OHLCV (daily)
- ✅ Basic stock info (market cap, P/E)
- ✅ 52-week high/low

### Available but Unused

#### Market Data
| Capability | Endpoint | Use Case |
|------------|----------|----------|
| **Real-time streaming** | `streamer.finance.yahoo.com` | Live price alerts |
| **Intraday data** | `history(interval="5m")` | Day trading analysis |
| **Extended hours** | `prepost=True` | After-hours trading |
| **Multi-ticker batch** | `download()` | Portfolio analysis |

#### Fundamental Data
| Capability | Endpoint | Use Case |
|------------|----------|----------|
| **Financial statements** | `financials`, `balance_sheet` | Value investing |
| **Cash flow** | `cashflow` | Dividend sustainability |
| **Valuation ratios** | `info` (PEG, EV/EBITDA) | Stock screening |
| **Dividend history** | `dividends` | Income investing |
| **Splits history** | `splits` | Adjusted backtests |

#### Options Data
| Capability | Endpoint | Use Case |
|------------|----------|----------|
| **Options chains** | `option_chain()` | Covered call strategies |
| **Implied volatility** | `option_chain()` | Volatility trading |
| **Open interest** | `option_chain()` | Unusual activity |

#### Corporate Actions
| Capability | Endpoint | Use Case |
|------------|----------|----------|
| **Earnings calendar** | `earnings_dates` | Event-driven trading |
| **Earnings history** | `earnings_history` | Growth analysis |
| **Upgrades/downgrades** | `recommendations` | Analyst consensus |

#### Screening & Discovery
| Capability | Endpoint | Use Case |
|------------|----------|----------|
| **Custom screeners** | `EquityQuery` | Find value stocks |
| **Sector performance** | `Sector` | Rotation strategies |
| **Market summary** | `market_summary` | Market health |

---

## Implementation Timeline

### Q1 2026
- Intraday support
- Batch analysis
- Excel export

### Q2 2026
- Backtesting engine
- Price alerts
- Fundamental screener

### Q3 2026
- Options module
- Earnings tracker
- Portfolio optimizer

### Q4 2026
- Advanced analytics
- Sentiment analysis
- Web interface (optional)

---

## Technical Considerations

### Rate Limits
- Yahoo Finance: ~2,000 requests/hour
- Recommended: Use caching aggressively
- Batch downloads preferred over individual calls

### Data Quality
- Real-time data delayed 15-20 minutes for some markets
- Not all stocks have options/fundamentals data
- Historical data may have gaps for delisted stocks

### Terms of Use
- Personal use only
- No redistribution
- Cache data locally
- Respect rate limits

---

## Contributing to Roadmap

To propose changes or additions:

1. Open an issue with label `roadmap`
2. Describe the feature and its value
3. Reference Yahoo Finance API capabilities
4. Provide use cases and examples

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

## Progress Tracking

**Completed:**
- [x] Core technical analysis (SMA, RSI, Bollinger Bands)
- [x] Support/Resistance detection
- [x] Volume Profile analysis
- [x] Chart generation with annotations
- [x] CLI interface with rich output
- [x] Configuration file support

**In Progress:**
- [ ] Documentation reorganization

**Planned:**
- See Phase 1-3 above

---

*Last updated: 2026-02-14*  
*Maintained by: IDX Stock Analyzer Team*
