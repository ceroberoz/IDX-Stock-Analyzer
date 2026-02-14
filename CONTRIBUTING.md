# Contributing to IDX Stock Analyzer

Thank you for your interest in contributing to IDX Stock Analyzer! This document provides guidelines and instructions for contributing.

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

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Follow the existing code style
- Add docstrings to functions and classes
- Update README.md if needed

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
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

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

## 🧪 Testing

### Manual Testing Checklist

Before submitting a PR, test:

- [ ] Basic analysis works: `uv run idx-analyzer BBCA`
- [ ] Chart generation works: `uv run idx-analyzer BBCA --chart`
- [ ] All periods work: `--period 1mo`, `--period 6mo`, `--period 1y`
- [ ] Export works: `--export csv`, `--export json`
- [ ] Quiet mode works: `--quiet`
- [ ] Error handling works (try invalid ticker like `AAPL`)
- [ ] Config creation works: `uv run idx-analyzer BBCA --init-config`
- [ ] Custom config works: `uv run idx-analyzer BBCA --config custom.toml`

### Test Stocks

Use these stocks for testing different scenarios:

- **BBCA** - Large cap, high volume
- **GOTO** - Tech stock, volatile
- **ADRO** - Mining sector
- **UNVR** - Consumer goods

## 📦 Project Structure

```
IDX-Stock-Analyzer/
├── idx_analyzer/          # Main package
│   ├── __init__.py
│   ├── analyzer.py        # Core analysis logic
│   ├── cli.py             # CLI interface
│   ├── config.py          # Configuration management
│   └── exceptions.py      # Custom exceptions
├── examples/              # Example scripts
├── tests/                 # Test files (future)
├── charts/                # Generated charts (gitignored)
├── docs/                  # Documentation
├── idx-analyzer.toml.example  # Example configuration
├── pyproject.toml         # Project config
└── README.md              # Main documentation
```

## 🎯 Areas for Contribution

### High Priority

- [ ] Unit tests with pytest
- [ ] Additional technical indicators (MACD, Fibonacci)
- [ ] Performance optimization

### Medium Priority

- [ ] Web interface (Flask/FastAPI)
- [ ] More export formats (Excel, PDF)
- [ ] Price alerts system

### Documentation

- [ ] Tutorial videos
- [ ] More examples
- [ ] API documentation
- [ ] Contributing translations

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

## 💡 Feature Requests

When suggesting features:

1. Check if it's already been suggested
2. Describe the use case
3. Explain why it would be useful
4. Provide examples if possible

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers get started

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## 📞 Questions?

- Open an issue for questions
- Join discussions
- Check existing documentation

Thank you for contributing! 🎉
