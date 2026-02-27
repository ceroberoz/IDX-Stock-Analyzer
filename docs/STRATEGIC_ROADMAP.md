# 🎯 IDX Stock Analyzer - Strategic Product Roadmap

> **Document Version:** 2.1  
> **Last Updated:** 2026-02-27  
> **Framework:** Scrum + User Personas  
> **Format:** Epics → Stories with Story Points

---

## 📊 Project Status Overview

| Epic | Status | Stories Done | Total Points |
|------|--------|--------------|--------------|
| Epic 1: Foundation & Stability | ✅ Complete | 3/3 | 18 pts |
| Epic 2: Enhanced Data & Charts | 🔄 In Progress | 1/3 | 13/34 pts |
| Epic 3: Fundamental Analysis | ⏳ Pending | 0/3 | 0/29 pts |
| Epic 4: Portfolio Management | ⏳ Pending | 0/4 | 0/54 pts |
| Epic 5: Backtesting Engine | ⏳ Pending | 0/4 | 0/60 pts |
| Epic 6: Advanced Screening & Alerts | ⏳ Pending | 0/3 | 0/29 pts |
| Epic 7: Institutional Data | ⏳ Pending | 0/3 | 0/29 pts |
| Epic 8: Real-Time & Streaming | ⏳ Pending | 0/2 | 0/21 pts |

**Total Progress:** 31/353 points (9%)

---

## 👥 USER PERSONAS

### Persona 1: "Budi - The Retail Investor"
- **Demographics:** 28-35 years old, works full-time, invests on weekends
- **Goals:** Find undervalued stocks, track portfolio performance, get alerts
- **Pain Points:** 
  - Doesn't have time to monitor markets during work hours
  - Confused by complex financial jargon
  - Wants simple BUY/SELL/HOLD recommendations
- **Tech Comfort:** Moderate (uses mobile apps, familiar with Excel)
- **Key Features Needed:** Simple watchlist, one-click reports, mobile-friendly UI

### Persona 2: "Sarah - The Day Trader"
- **Demographics:** 30-40 years old, trades full-time or as side income
- **Goals:** Identify entry/exit points, monitor multiple timeframes, backtest strategies
- **Pain Points:**
  - Needs real-time data with minimal delay
  - Wants to test strategies before risking capital
  - Requires technical indicators on multiple timeframes
- **Tech Comfort:** High (uses Bloomberg Terminal, trading platforms)
- **Key Features Needed:** Intraday charts (1m, 5m, 15m), backtesting engine, real-time alerts

### Persona 3: "Mr. Hartono - The Value Investor"
- **Demographics:** 45-60 years old, high net worth, long-term investor
- **Goals:** Find quality companies, analyze fundamentals, diversify portfolio
- **Pain Points:**
  - Needs comprehensive financial statement analysis
  - Wants to compare companies across sectors
  - Requires portfolio optimization tools
- **Tech Comfort:** Moderate (uses desktop applications)
- **Key Features Needed:** Financial statements, valuation ratios (P/E, P/B), PDF reports

### Persona 4: "Analyst Dina - The Financial Professional"
- **Demographics:** 25-35 years old, works at investment firm or bank
- **Goals:** Research stocks for clients, generate professional reports, track institutional activity
- **Tech Comfort:** Very High (uses Python, APIs, Bloomberg)
- **Key Features Needed:** Institutional ownership data, insider tracking, REST API

---

## ✅ COMPLETED EPICS

### Epic 1: Foundation & Stability ✅
**Status:** COMPLETE (Week 1-4) | **Points:** 18/18

#### Story 1.1: Clean Production Code ✅
- [x] DEBUG statements removed from `market_screen.py`
- [x] All 16 silent `except: pass` blocks replaced with proper logging
- [x] No console noise in production builds
- **Story Points:** 5 | **Priority:** P0

#### Story 1.2: SQLite Caching ✅
- [x] TUI cache implemented with SQLite (`tui_cache.py`)
- [x] Cache TTL configurable
- [x] Cache invalidation on demand
- **Story Points:** 5 | **Priority:** P1

#### Story 1.3: Excel Export Feature ✅
- [x] Export includes Summary, Technicals, Fundamentals sheets
- [x] Formatted headers and conditional formatting (green/red)
- [x] Support for `--export excel` CLI flag
- **Story Points:** 8 | **Priority:** P1

---

## 🔄 IN PROGRESS

### Epic 2: Enhanced Data & Charts 🔄
**Status:** IN PROGRESS (Sprint 3-5) | **Points:** 13/34 | **Target Personas:** Sarah, Budi

#### Story 2.1: Intraday Analysis Support ✅
- [x] Support intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
- [x] Respect Yahoo's limits (1m=7d, 5-30m=1mo, 1h=3mo)
- [x] `--interval` CLI flag works with all periods
- [x] TUI screen shows interval selector
- **Story Points:** 13 | **Priority:** P1

#### Story 2.2: Interactive TUI Charts ⏳
- [ ] Charts render using `textual-plotext` inside TUI
- [ ] Keyboard navigation: arrow keys for pan, +/- for zoom
- [ ] Toggle indicators with hotkeys (R for RSI, M for MACD)
- **Story Points:** 13 | **Priority:** P1

#### Story 2.3: Multi-Ticker Comparison Charts ⏳
- [ ] Compare up to 4 tickers normalized by % change
- [ ] Batch download with `yf.download()` for efficiency
- [ ] Legend showing ticker symbols
- **Story Points:** 8 | **Priority:** P2

---

## ⏳ PENDING EPICS

### Epic 3: Fundamental Analysis ⏳
**Status:** PENDING (Sprint 6-8) | **Points:** 29 | **Target Personas:** Mr. Hartono, Analyst Dina

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 3.1: Financial Statements View | 13 | P1 | ⏳ |
| 3.2: Valuation Dashboard | 8 | P1 | ⏳ |
| 3.3: Financial Health Scoring | 8 | P2 | ⏳ |

---

### Epic 4: Portfolio Management ⏳
**Status:** PENDING (Sprint 9-12) | **Points:** 54 | **Target Personas:** Mr. Hartono, Budi

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 4.1: Portfolio Data Model | 8 | P0 | ⏳ |
| 4.2: Portfolio Dashboard | 13 | P1 | ⏳ |
| 4.3: Risk Metrics Calculation | 8 | P1 | ⏳ |
| 4.4: Portfolio Optimization | 13 | P2 | ⏳ |

---

### Epic 5: Backtesting Engine ⏳
**Status:** PENDING (Sprint 13-16) | **Points:** 60 | **Target Personas:** Sarah, Analyst Dina

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 5.1: Strategy Definition DSL | 13 | P1 | ⏳ |
| 5.2: Backtest Engine Core | 21 | P1 | ⏳ |
| 5.3: Backtest Results Dashboard | 13 | P2 | ⏳ |
| 5.4: Walk-Forward Analysis | 13 | P3 | ⏳ |

---

### Epic 6: Advanced Screening & Alerts ⏳
**Status:** PENDING (Sprint 17-19) | **Points:** 29 | **Target Personas:** Sarah, Budi

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 6.1: Fundamental Screener Filters | 8 | P1 | ⏳ |
| 6.2: Price & Technical Alerts | 13 | P1 | ⏳ |
| 6.3: Telegram/Webhook Notifications | 8 | P2 | ⏳ |

---

### Epic 7: Institutional Data ⏳
**Status:** PENDING (Sprint 20-22) | **Points:** 29 | **Target Personas:** Analyst Dina, Mr. Hartono

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 7.1: Institutional Ownership Data | 8 | P2 | ⏳ |
| 7.2: Insider Transaction Tracking | 8 | P2 | ⏳ |
| 7.3: Options Chain Viewer | 13 | P2 | ⏳ |

---

### Epic 8: Real-Time & Streaming ⏳
**Status:** PENDING (Sprint 23-24) | **Points:** 21 | **Target Personas:** Sarah

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| 8.1: WebSocket Live Streaming | 13 | P1 | ⏳ |
| 8.2: Real-Time Market Dashboard | 8 | P2 | ⏳ |

---

## ❌ DEFERRED / OUT OF SCOPE

| Item | Reason |
|------|--------|
| Web Dashboard (Phase 4) | TUI is core strength; web is nice-to-have |
| Mobile App | Out of scope; responsive web is sufficient |
| AI Stock Prediction | Unreliable; focus on factual data |
| Social Media Sentiment | Low accuracy; focus on news |
| Cryptocurrency Support | Different market; out of scope |

---

## 📅 SPRINT PLANNING

| Sprint | Focus | Status | Points |
|--------|-------|--------|--------|
| Sprint 1 | Foundation - Error handling | ✅ Complete | 5 |
| Sprint 2 | Excel Export | ✅ Complete | 8 |
| Sprint 3 | Intraday Analysis | ✅ Complete | 13 |
| Sprint 4 | TUI Charts | 🔄 In Progress | 13 |
| Sprint 5 | Multi-Ticker Comparison | ⏳ Planned | 8 |
| Sprint 6-8 | Fundamental Analysis | ⏳ Planned | 29 |
| Sprint 9-12 | Portfolio Management | ⏳ Planned | 54 |
| Sprint 13-16 | Backtesting Engine | ⏳ Planned | 60 |
| Sprint 17-19 | Screening & Alerts | ⏳ Planned | 29 |
| Sprint 20-22 | Institutional Data | ⏳ Planned | 29 |
| Sprint 23-24 | Real-Time Streaming | ⏳ Planned | 21 |

**Total Duration:** 48 weeks (~11 months)  
**Total Story Points:** 353 points  
**Velocity Assumption:** 15-20 points per sprint (2-person team)

---

## 🎯 DEFINITION OF DONE

### For Each Story:
- [x] Code complete and follows project style guide
- [ ] Unit tests written (minimum 80% coverage) ⏳
- [ ] Integration tests pass ⏳
- [x] Documentation updated (README, API docs)
- [x] Code review completed
- [x] No TODOs or FIXMEs in code
- [x] Performance tested (no regressions)
- [x] User acceptance criteria met

---

## 📊 SUCCESS METRICS (KPIs)

| Metric | Target (6mo) | Target (12mo) |
|--------|--------------|---------------|
| Daily Active Users | 100 | 500 |
| Avg Session Duration | 10 min | 15 min |
| Reports Exported | 200/month | 1000/month |
| Test Coverage | > 80% | > 85% |
| GitHub Stars | 100 | 500+ |

---

## 🔗 RELATED DOCUMENTS

- [📋 Detailed Roadmap](./ROADMAP.md) - Technical implementation details
- [📊 Bloomberg Mapping](./BLOOMBERG_ROADMAP.md) - Feature mapping to Bloomberg Terminal
- [📚 User Guide](./USAGE.md) - Complete command reference
- [💻 API Reference](./API.md) - Developer documentation
- [🤝 Contributing](../../CONTRIBUTING.md) - Contribution guidelines
- [🗺️ Implementation Status](./IMPLEMENTATION_STATUS.md) - Detailed feature status

---

*This roadmap is a living document. Update monthly based on user feedback and market conditions.*
