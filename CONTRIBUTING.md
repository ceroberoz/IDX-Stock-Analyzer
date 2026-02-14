# Contributing to IDX Stock Analyzer

Thank you for your interest in contributing to IDX Stock Analyzer! This document provides guidelines and instructions for contributing.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- UV package manager
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/ceroberoz/IDX-Stock-Analyzer.git
cd IDX-Stock-Analyzer

# Install dependencies
uv sync

# Verify setup
uv run idx-analyzer --version
```

---

## 🌿 Branching Strategy

We use a multi-branch workflow:

```
main (production-ready)
  └── development (integration branch)
        └── research (experimental features)
        └── feature/* (new features)
        └── fix/* (bug fixes)
```

### Branch Types

| Branch | Purpose | Base | Merge Target |
|--------|---------|------|--------------|
| `main` | Production releases | - | - |
| `development` | Integration branch | `main` | `main` |
| `research` | R&D, documentation | `development` | `development` |
| `feature/*` | New features | `development` | `development` |
| `fix/*` | Bug fixes | `development` | `development` |

### Creating a Branch

```bash
# For new features
git checkout development
git checkout -b feature/your-feature-name

# For bug fixes
git checkout development
git checkout -b fix/issue-description

# For research/documentation
git checkout development
git checkout -b research/topic-name
```

---

## 📋 Development Workflow

### 1. Pick the Right Branch

Check the [Roadmap](docs/ROADMAP.md) to see which phase aligns with your contribution:

- **Phase 1 (Core Enhancements)**: Intraday, batch analysis, Excel export
- **Phase 2 (Advanced Analytics)**: Backtesting, alerts, screeners
- **Phase 3 (Institutional)**: Portfolio optimization, sentiment analysis

### 2. Make Changes

- Follow the existing code style
- Add docstrings to functions and classes
- Update relevant documentation in `docs/`

### 3. Format and Lint

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix
```

### 4. Test Your Changes

```bash
# Run the tool
uv run idx-analyzer BBCA --chart

# Test different periods
uv run idx-analyzer BBCA --period 1y

# Test export functionality
uv run idx-analyzer BBCA --export json
```

### 5. Commit

```bash
git add .
git commit -m "feat: add your feature description"
```

**Commit Message Convention:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 6. Push and Create PR

```bash
# Push to your branch
git push origin feature/your-feature-name

# Create PR to development branch (not main!)
```

**PR Guidelines:**
- Target the `development` branch (not `main`)
- Reference related issues
- Include test results
- Update documentation if needed

---

## 🎯 Development Phases

Based on Yahoo Finance API capabilities, development is organized in phases:

### Phase 1: Core Enhancements (Immediate)

**Goal:** Expand current capabilities with low-effort, high-value features

| Feature | Yahoo Finance API | Difficulty | Status |
|---------|------------------|------------|--------|
| Intraday Analysis | `history(interval="15m")` | Easy | Open |
| Batch Portfolio | `yf.download()` | Easy | Open |
| Dividend Calendar | `Ticker.dividends` | Easy | Open |
| Excel Export | Current + `pd.ExcelWriter` | Easy | Open |
| Stock Splits | `Ticker.splits` | Easy | Open |

**Good for:** New contributors, first PRs

### Phase 2: Advanced Analytics (Medium-Term)

**Goal:** Implement sophisticated analysis features

| Feature | Yahoo Finance API | Difficulty | Status |
|---------|------------------|------------|--------|
| Backtesting Engine | `history()` historical | Medium | Open |
| Real-time Alerts | WebSocket streaming | Medium | Open |
| Fundamental Screener | `screener.EquityQuery` | Medium | Open |
| Options Flow | `option_chain()` | Hard | Open |
| Earnings Tracker | `earnings_dates` | Medium | Open |

**Good for:** Experienced contributors, complex features

### Phase 3: Institutional-Grade (Advanced)

**Goal:** Professional-level tools

| Feature | Yahoo Finance API | Difficulty | Status |
|---------|------------------|------------|--------|
| Portfolio Optimization | Multi-ticker correlation | Hard | Open |
| Volatility Surface | Options across expirations | Hard | Open |
| Insider Tracking | Holders data | Medium | Open |
| News Sentiment | `Ticker.news` | Medium | Open |
| Analyst Ratings | `recommendations` | Easy | Open |

**Good for:** Domain experts, research contributions

---

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Example

```python
def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price data series
        window: Number of periods
        
    Returns:
        SMA series
    """
    return data.rolling(window=window).mean()
```

---

## 🧪 Testing

### Manual Testing Checklist

Before submitting a PR, test:

- [ ] Basic analysis works: `uv run idx-analyzer BBCA`
- [ ] Chart generation works: `uv run idx-analyzer BBCA --chart`
- [ ] All periods work: `--period 1mo`, `--period 6mo`, `--period 1y`
- [ ] Export works: `--export csv`, `--export json`
- [ ] Chat output works: `--chat`
- [ ] Quiet mode works: `--quiet`
- [ ] Error handling works (try invalid ticker like `AAPL`)
- [ ] Config creation works: `uv run idx-analyzer BBCA --init-config`
- [ ] Custom config works: `uv run idx-analyzer BBCA --config custom.toml`
- [ ] Chart outputs to `charts/TICKER/DATE/`: `uv run idx-analyzer BBCA --chart` then check `charts/BBCA/`
- [ ] Export outputs to `exports/TICKER/DATE/`: `uv run idx-analyzer BBCA --export json` then check `exports/BBCA/`

### Test Stocks

Use these stocks for testing different scenarios:

- **BBCA** - Large cap, high volume
- **GOTO** - Tech stock, volatile
- **ADRO** - Mining sector
- **UNVR** - Consumer goods

---

## 📁 Project Structure

```
IDX-Stock-Analyzer/
├── idx_analyzer/              # Main package
│   ├── __init__.py
│   ├── analyzer.py            # Core analysis engine
│   ├── cache.py               # HTTP cache management
│   ├── cli.py                 # CLI interface
│   ├── config.py              # Configuration management
│   ├── exceptions.py          # Custom exceptions
│   └── sentiment.py           # News sentiment analysis
├── charts/                    # Generated charts (gitignored content)
├── exports/                   # Generated exports (gitignored content)
├── docs/                      # Documentation
│   ├── API.md                 # Python API reference
│   ├── ROADMAP.md             # Development roadmap
│   ├── TECHNICAL_ANALYSIS.md  # TA explanations
│   └── USAGE.md               # User guide
├── examples/                  # Example scripts
├── tests/                     # Test files
├── pyproject.toml             # Project & build configuration
├── uv.lock                    # Dependency lock file
├── .python-version            # Python version pin
├── README.md                  # Main documentation
├── CONTRIBUTING.md            # Contribution guidelines
└── LICENSE                    # MIT License
```

---

## 🎁 Areas for Contribution

### High Priority (Phase 1)

- [ ] Unit tests with pytest
- [ ] Intraday data support (`--interval` flag)
- [ ] Batch analysis for multiple tickers
- [ ] Excel export format
- [ ] Dividend tracking module

### Medium Priority (Phase 2)

- [ ] Backtesting module
- [ ] Price alerts system (email/webhook)
- [ ] Fundamental screener
- [ ] Web interface (Flask/FastAPI)

### Documentation

- [ ] Tutorial videos
- [ ] More examples in `examples/`
- [ ] API documentation improvements
- [ ] Contributing translations

### Research (Phase 3)

- [ ] Portfolio optimization algorithms
- [ ] Sentiment analysis integration
- [ ] Options analysis tools

---

## 🔬 Yahoo Finance API Research

We use `yfinance` library which wraps Yahoo Finance API. Available capabilities:

### Market Data
- Real-time quotes (15-20 min delayed)
- Historical OHLCV data
- Intraday data (1m, 5m, 15m, 30m, 1h)
- Options chains with Greeks

### Fundamental Data
- Financial statements (income, balance, cash flow)
- Valuation ratios (P/E, PEG, EV/EBITDA)
- Dividend history and splits
- Company info and sector data

### Corporate Actions
- Earnings dates and history
- Analyst recommendations
- Insider transactions
- Institutional holders

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed API capabilities mapping.

---

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description** - Clear description of the bug
2. **Steps to Reproduce** - How to trigger the bug
3. **Expected Behavior** - What should happen
4. **Actual Behavior** - What actually happens
5. **Environment**:
   - Python version: `python --version`
   - OS: macOS/Linux/Windows
   - Tool version: `uv run idx-analyzer --version`

### Example Bug Report

```markdown
**Bug**: Chart not generating for certain tickers

**Steps to Reproduce**:
1. Run `uv run idx-analyzer GOTO --chart`
2. Observe error

**Expected**: Chart generated successfully

**Actual**: Error "Could not fetch data"

**Environment**:
- Python 3.13.1
- macOS 14.2
- idx-analyzer 0.1.0
```

---

## 💡 Feature Requests

When suggesting features:

1. Check the [Roadmap](docs/ROADMAP.md) if it's already planned
2. Describe the use case
3. Explain why it would be useful
4. Reference Yahoo Finance API capabilities if applicable
5. Provide examples if possible

---

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers get started

---

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

---

## 📞 Questions?

- Open an issue with label `question`
- Join GitHub Discussions
- Check existing documentation in `docs/`

Thank you for contributing! 🎉
