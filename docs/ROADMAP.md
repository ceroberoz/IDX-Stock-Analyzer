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
| **Sentiment Analysis** | `Ticker.news` + FinBERT | 🚧 In Progress | High |
| **Dividend Calendar** | `Ticker.dividends` | Planned | Medium |
| **Stock Splits** | `Ticker.splits` | Planned | Medium |
| **Excel Export** | Current data + `pd.ExcelWriter` | Planned | Medium |

**Deliverables:**
- [ ] Add `--interval` flag (1m, 5m, 15m, 30m, 1h, 1d)
- [ ] Multi-ticker analysis: `idx-analyzer BBCA BBRI TLKM`
- [ ] Sentiment analysis module (Yahoo news + FinBERT) - ✅ **No API key required**
- [ ] `--sentiment` CLI flag for news sentiment scoring
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

## Alternative Data Sources

Beyond Yahoo Finance, several alternative data sources are available for IDX (Indonesia Stock Exchange) data:

### Official Sources
| Source | Free Tier | Data | Best For |
|--------|-----------|------|----------|
| **BEI/IDX Official** | ❌ Commercial | Real-time, EOD, historical, corporate actions | Production, institutional use |
| **IDX Data Download** | ✅ With registration | Market summary, trading data, foreign ownership | Official EOD data |

### Commercial APIs
| Source | Free Tier | Price | Data Coverage |
|--------|-----------|-------|---------------|
| **OHLC.dev** | ✅ 500/mo | $15/mo | 50+ endpoints, stocks, bonds, derivatives |
| **Finnhub** | ✅ 60/min | Free tier | Global + IDX, includes news sentiment |
| **GoAPI.io** | ✅ Trial | Affordable | Real-time prices, volume, indices |
| **Sectors.app** | ❌ | $49/mo | Indonesia-focused, fundamentals |
| **Alpha Vantage** | ✅ 25/day | Free tier | Global stocks, news, sentiment |

### Open Source Projects (FREE)
- **risan/indonesia-stock-exchange** - Pasardana scraper, deploy to Vercel
- **noczero/idx-fundamental-analysis** - StockBit + YFinance, active development
- **alifianmahardhika/idx-scraper** - Selenium-based, monthly updates

### Kaggle Datasets (FREE)
- Indonesia Stock Market 2020-2024 (with foreign flow data)
- LQ45 Stocks Historical (Oct 2023-Oct 2024)
- Individual bank stocks (BBCA, BBRI, BMRI)

---

## Sentiment Analysis Implementation

### Current Approach: Yahoo Finance + FinBERT (FREE)

**Architecture:**
```
Yahoo Finance (news) → FinBERT (NLP model) → Sentiment Score
```

**Advantages:**
- ✅ Uses existing Yahoo Finance dependency
- ✅ No API registration required
- ✅ No rate limits or usage fees
- ✅ Runs locally (privacy)
- ✅ Financial domain-specific model (FinBERT)

**Implementation:**
```python
# Get news from Yahoo Finance
news = ticker.news  # Returns title, publisher, link, date

# Analyze with FinBERT
from transformers import pipeline
finbert = pipeline("sentiment-analysis", model="ProsusAI/finBERT")
sentiment = finbert(news_title)[0]
# Returns: {'label': 'positive', 'score': 0.987}
```

**Output:**
- Aggregate sentiment score (-1 to +1)
- Positive/Negative/Neutral breakdown
- Confidence scores per article
- Recent news headlines with sentiment

### Alternative: Third-Party APIs

| Source | Free Tier | Registration | Pros | Cons |
|--------|-----------|--------------|------|------|
| **Finnhub** | 60/min | ✅ Required | Built-in sentiment | API key needed |
| **marketaux** | 100/day | ✅ Required | Built-in sentiment | Limited requests |
| **Sectors.app** | ❌ | ✅ Required | Indonesia-focused | Paid ($49/mo) |

**Decision:** Use **Yahoo + FinBERT** for Phase 1 (no registration, free forever)

---

## Sentiment Analysis - Current Status & Future Enhancements

### Current Implementation Status: ✅ FUNCTIONAL BUT LIMITED

**What Works:**
- Sentiment analysis module implemented (`idx_analyzer/sentiment.py`)
- Both FinBERT and VADER models supported
- CLI commands: `--sentiment` and `--sentiment-vader`
- Proper error handling for missing dependencies

**Current Limitations:**
- **Limited News Coverage**: Yahoo Finance provides minimal news for Indonesian stocks
- **English-Only Analysis**: FinBERT/VADER work best with English text; Indonesian financial news often in Bahasa Indonesia
- **Small Dataset**: Testing shows 0-5 articles per ticker vs 20-50 for US stocks
- **Model Download**: FinBERT requires ~500MB download on first run

### Future Enhancement Recommendations

#### Phase 1A: Immediate Improvements (Low Effort)

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Local News Scrapers** | Medium | High | Scrape Detik Finance, Kontan, Investor Daily Indonesia |
| **Multilingual Support** | Medium | High | Add Indonesian NLP model (e.g., IndoBERT) |
| **Sentiment Cache** | Low | Medium | Cache sentiment results to reduce API calls |
| **Batch Sentiment** | Low | Medium | Analyze multiple tickers in one command |

#### Phase 2A: Advanced Sentiment (Medium Effort)

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Twitter/X Integration** | Medium | High | Social media sentiment for trending stocks |
| **Reddit r/IndoFinance** | Low | Medium | Community sentiment from local investors |
| **StockTwits API** | Low | Medium | Trader sentiment platform |
| **Sectors.app Fallback** | Low | High | Use Sectors.app for Indonesia-specific news |

#### Phase 3A: AI-Powered Sentiment (High Effort)

| Enhancement | Effort | Impact | Description |
|-------------|--------|--------|-------------|
| **Custom FinBERT Training** | High | Very High | Train on Indonesian financial corpus |
| **LLM Integration** | Medium | High | Use GPT/Claude for advanced sentiment + reasoning |
| **Sentiment Trends** | Medium | Medium | Track sentiment changes over time |
| **Correlation Analysis** | Medium | High | Correlate sentiment with price movements |

### Data Source Recommendations

**For Better Indonesian Stock Sentiment:**

1. **Primary**: Yahoo Finance (current) - Free, easy, limited coverage
2. **Secondary**: Local News Scrapers
   - Detik Finance: https://finance.detik.com/
   - Kontan: https://kontan.co.id/
   - Investor Daily: https://investor.id/
3. **Tertiary**: Social Media
   - Twitter/X API (academic/research access)
   - Reddit r/IndoFinance
   - StockTwits
4. **Premium**: Sectors.app ($49/mo) - Indonesia-focused with better coverage

### Implementation Priority

**Next 2 Weeks:**
- [ ] Add documentation about sentiment limitations
- [ ] Implement news scraper for Detik Finance
- [ ] Add fallback when Yahoo returns no news

**Next Month:**
- [ ] Integrate IndoBERT for Bahasa Indonesia support
- [ ] Add Twitter/X sentiment (if API available)
- [ ] Cache sentiment results in SQLite

**Next Quarter:**
- [ ] Custom model training on Indonesian financial corpus
- [ ] Sentiment trend visualization on charts
- [ ] Real-time sentiment alerts

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
- Sentiment analysis enhancements (see Phase 1A-3A above)
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
- [x] Documentation reorganization
- [ ] Sentiment analysis module (Yahoo + FinBERT)

**Planned:**
- See Phase 1-3 above

---

*Last updated: 2026-02-15*  
*Maintained by: IDX Stock Analyzer Team*
